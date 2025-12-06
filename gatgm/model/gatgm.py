from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from gatgm.data import GetPubChemFPs, create_graph, get_atom_features_dim
import csv

atts_out = []

class FPN(nn.Module):
    def __init__(self,args):
        super(FPN, self).__init__()
        self.fp_2_dim=args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args
        if hasattr(args,'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'
        
        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024
        
        if hasattr(args,'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None
        
        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)
    
    def forward(self, smile):
        fp_list=[]
        for i, one in enumerate(smile):
            fp=[]
            mol = Chem.MolFromSmiles(one)
            
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
                fp_pubcfp = GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)
                
        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:,self.fp_changebit-1] = np.ones(fp_list[:,self.fp_changebit-1].shape)
            fp_list.tolist()
        
        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_gnn, alpha, inter_graph, concat=True):
        super(GATLayer, self).__init__()
        self.dropout_gnn= dropout_gnn
        self.in_features = in_features 
        self.out_features = out_features
        self.alpha = alpha 
        self.concat = concat 
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        #边特征线性层
        self.edge_dim = 20  # 假设edge_dim=20，可以从args传入
        self.edge_linear = nn.Linear(self.edge_dim, 1)

    def _prepare_attentional_input(self, Wh):
        """
        Wh: tensor (N, out_features)
        返回 a_input: tensor (N, N, 2*out_features)，
        其中 a_input[i,j] = concat(Wh[i], Wh[j])
        """
        N = Wh.size(0)                # 节点数
        # Wh.unsqueeze(1): (N,1,out_features) -> repeat -> (N,N,out_features)
        Wh_repeated_in_chunks = Wh.unsqueeze(1).repeat(1, N, 1)
        # Wh.unsqueeze(0): (1,N,out_features) -> repeat -> (N,N,out_features)
        Wh_repeated_alternating = Wh.unsqueeze(0).repeat(N, 1, 1)
        a_input = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)  # (N,N,2*out_features)
        return a_input
        
    def forward(self, mole_out, adj, edge_feat=None):
        Wh = self.W(mole_out)
        a_input = self._prepare_attentional_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        
        # 3D边特征影响注意力分数
        if edge_feat is not None:
            edge_score = self.edge_linear(edge_feat).squeeze(-1)
            e = e + edge_score
        
        # 确保 zero_vec 在与 e 相同的设备上
        zero_vec = -9e15 * torch.ones_like(e).to(e.device) # <--- 设备兼容
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.Softmax(dim=1)(attention)
        attention = nn.Dropout(p=self.dropout_gnn)(attention)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return nn.ELU()(h_prime)
        else:
            return h_prime


class GATOne(nn.Module):
    def __init__(self,args):
        super(GATOne, self).__init__()
        self.nfeat = get_atom_features_dim()
        self.nhid = args.nhid
        self.dropout_gnn = args.dropout_gat
        self.atom_dim = args.hidden_size
        self.alpha = 0.2
        self.nheads = args.nheads
        self.args = args
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        
        if hasattr(args,'inter_graph'):
            self.inter_graph = args.inter_graph
        else:
            self.inter_graph = None
        
        self.attentions = [GATLayer(self.nfeat, self.nhid, dropout_gnn=self.dropout_gnn, alpha=self.alpha, inter_graph=self.inter_graph, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(self.nhid * self.nheads, self.atom_dim, dropout_gnn=self.dropout_gnn, alpha=self.alpha, inter_graph=self.inter_graph, concat=False)

    def forward(self, mole_out, adj, edge_feat=None):
        mole_out = self.dropout(mole_out)
        mole_out = torch.cat([att(mole_out, adj, edge_feat=edge_feat) for att in self.attentions], dim=1)
        mole_out = self.dropout(mole_out)
        mole_out = nn.functional.elu(self.out_att(mole_out, adj, edge_feat=edge_feat))
        return nn.functional.log_softmax(mole_out, dim=1)

class GATEncoder(nn.Module):
    def __init__(self,args):
        super(GATEncoder,self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GATOne(self.args)
    
    def forward(self, mols, smiles, edge_feat=None):
        atom_feature, atom_index = mols.get_feature()
        device = next(self.parameters()).device 

        if self.cuda and device.type == 'cuda':
            atom_feature = atom_feature.to(device) 

        gat_outs=[]
        for i,one in enumerate(smiles):
            # ... (省略 adj 和 one_feature 的生成，它们已经在 CPU/CUDA 上)
            adj = []
            mol = Chem.MolFromSmiles(one)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj/1
            adj = torch.from_numpy(adj).to(device)

            atom_start, atom_size = atom_index[i]
            one_feature = atom_feature[atom_start:atom_start+atom_size]
            
            # 确保 edge_feat_i 在 if 块外被定义
            edge_feat_i = None 
            
            # 初始化 edge_feat_tensor 以避免 UnboundLocalError
            edge_feat_tensor = None # <--- 修复: 提前初始化

            if edge_feat is not None:
                # edge_feat 应该是一个列表/元组，其中包含批次中所有分子的边特征
                if isinstance(edge_feat, (list, tuple)) and i < len(edge_feat):
                    edge_feat_tensor = edge_feat[i]
                
                # 2. **核心修改：将 edge_feat_tensor 移动到正确的设备**
                if edge_feat_tensor is not None: # 使用初始化后的变量
                    # 将 edge_feat_tensor 移动到与模型相同的设备
                    edge_feat_tensor = edge_feat_tensor.to(device) 

                    if torch.count_nonzero(edge_feat_tensor) == 0:
                        print("X", end="")
                    
                    N_heavy = atom_size
                    N_full = edge_feat_tensor.size(0)
                    
                    if N_full != N_heavy:
                        print(">",end="")
                        # 假设 edge_feat_tensor 是 (N_full, N_full, 20)
                        # 我们需要切片到 (N_heavy, N_heavy, 20)
                        edge_feat_i = edge_feat_tensor[:N_heavy, :N_heavy, :]
                    else:
                        print("^",end="")
                        edge_feat_i = edge_feat_tensor

            gat_atoms_out = self.encoder(one_feature, adj, edge_feat=edge_feat_i)
            gat_out = gat_atoms_out.sum(dim=0)/atom_size
            gat_outs.append(gat_out)
        
        gat_outs = torch.stack(gat_outs, dim=0)
        return gat_outs

class GAT(nn.Module):
    def __init__(self,args):
        super(GAT,self).__init__()
        self.args = args
        self.encoder = GATEncoder(self.args)
        
    def forward(self, smile, edge_feat=None):
        mol = create_graph(smile, self.args)
        gat_out = self.encoder.forward(mol, smile, edge_feat=edge_feat)
        return gat_out



class FpgnnModel(nn.Module):
    def __init__(self,is_classif,gat_scale,cuda,dropout_fpn, args): # 接收 args
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        self.args = args # 存储 args
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self,args):
        self.encoder3 = GAT(args)
    
    def create_fpn(self,args):
        self.encoder2 = FPN(args)
    
    def create_scale(self,args):
        linear_dim = int(args.hidden_size)
        if self.gat_scale == 1:
            self.fc_gat = nn.Linear(linear_dim,linear_dim)
        elif self.gat_scale == 0:
            self.fc_fpn = nn.Linear(linear_dim,linear_dim)
        else:
            self.gat_dim = int((linear_dim*2*self.gat_scale)//1)
            self.fc_gat = nn.Linear(linear_dim,self.gat_dim)
            self.fc_fpn = nn.Linear(linear_dim,linear_dim*2-self.gat_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self,args):
        linear_dim = args.hidden_size
        if self.gat_scale == 1:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )
        elif self.gat_scale == 0:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )

        else:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim*2, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )
    
    def forward(self, input, edge_feat=None):
        """
        input: list of SMILES (batch)
        edge_feat: None or
                   - list of tensors, each tensor shape (n_i, n_i, edge_dim) per molecule in batch, or
                   - single-element list [tensor] if batch size == 1
        This implementation runs GAT per-molecule (loop) to avoid treating the batch as one huge graph.
        """
        # CASE gat only: run GAT per molecule and return stacked outputs
        if self.gat_scale == 1:
            gat_outs = []
            for i, sm in enumerate(input):
                # prepare per-molecule edge feature argument for encoder3
                ef_i = None
                if edge_feat is not None:
                    # edge_feat expected as list-like where edge_feat[i] is (n_i, n_i, edge_dim)
                    if isinstance(edge_feat, (list, tuple)):
                        ef_i = [edge_feat[i]]  # keep as list for create_graph semantics
                    elif torch.is_tensor(edge_feat):
                        # if a tensor with first dim == batch (unlikely in current pipeline),
                        # try to index it
                        try:
                            ef_i = [edge_feat[i].detach()]  # convert to single-element list
                        except Exception:
                            ef_i = None
                    else:
                        ef_i = None
                # encoder3 expects (smiles_list, edge_feat=list_or_None)
                out = self.encoder3([sm], edge_feat=ef_i)  # out shape: (1, hidden)
                gat_outs.append(out)
            gat_out = torch.cat(gat_outs, dim=0)  # (batch, hidden)
            output = gat_out

        # CASE fpn only: encoder2 already processes batch (list) and returns (batch, hidden)
        elif self.gat_scale == 0:
            output = self.encoder2(input)

        # CASE fusion: run GAT per-molecule (loop) and FPN on whole batch, then combine
        else:
            # GAT per-molecule
            gat_outs = []
            for i, sm in enumerate(input):
                ef_i = None
                if edge_feat is not None:
                    if isinstance(edge_feat, (list, tuple)):
                        ef_i = [edge_feat[i]]
                    elif torch.is_tensor(edge_feat):
                        try:
                            ef_i = [edge_feat[i].detach()]
                        except Exception:
                            ef_i = None
                    else:
                        ef_i = None
                out = self.encoder3([sm], edge_feat=ef_i)  # (1, hidden)
                gat_outs.append(out)
            gat_out = torch.cat(gat_outs, dim=0)  # (batch, hidden)

            # FPN processes batch at once
            fpn_out = self.encoder2(input)  # should be (batch, hidden)

            # project and activate
            gat_out = self.fc_gat(gat_out)
            gat_out = self.act_func(gat_out)
            fpn_out = self.fc_fpn(fpn_out)
            fpn_out = self.act_func(fpn_out)

            output = torch.cat([gat_out, fpn_out], dim=1)  # (batch, hidden*2*scale)

        # feed to final FFN
        output = self.ffn(output)

        # if classification and evaluation mode, sigmoid the outputs
        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output


def get_atts_out():
    return atts_out

def FPGNN(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    # 传递 args 给 FpgnnModel
    model = FpgnnModel(is_classif,args.gat_scale,args.cuda,args.dropout, args) 
    if args.gat_scale == 1:
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_fpn(args)
        model.create_scale(args)
        model.create_ffn(args)
    
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    return model