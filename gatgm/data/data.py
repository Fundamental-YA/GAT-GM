from argparse import Namespace
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig
import random
import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
import traceback

class MoleData:
    def __init__(self, line, args):
        self.args = args
        self.smile = line[0]
        self.mol = Chem.MolFromSmiles(self.smile)
        self.label = [float(x) if x != '' else None for x in line[1:]]
        self.dihedral_angles = []
        self.pairwise_distances = []
        self.centroid_distances = []
        self.steric_hindrance = []
        self.molecular_volume = None
        self.hbond_donor_acceptor_distances = []
        self.edge_feat = None

        if hasattr(args, 'use_3d_features') and args.use_3d_features and self.mol is not None:
            try:
                self.mol = Chem.AddHs(self.mol)
                
                # --- 尝试 1: 默认的 K-M 算法 (EmbedMolecule) ---
                if AllChem.EmbedMolecule(self.mol, randomSeed=42) == 0:
                    AllChem.MMFFOptimizeMolecule(self.mol)
                    self._extract_3d_features()
                else:
                    # --- 尝试 2: 失败后，使用更稳健的 AllChem.GenerateDepiction (2D) 或更激进的 EmbedGenerator ---
                    print(f"警告: EmbedMolecule失败，尝试回退策略... SMILES: {self.smile}")
                    
                    # 尝试用更激进的参数再次生成构象
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    
                    if AllChem.EmbedMolecule(self.mol, params) == 0:
                        AllChem.MMFFOptimizeMolecule(self.mol)
                        self._extract_3d_features()
                    else:
                        print(f"严重警告: 3D构象生成彻底失败。SMILES: {self.smile}")
                        if self.mol:
                            self.mol = Chem.RemoveHs(self.mol)
                        self.edge_feat = None # 保持 None，等待数据集处理
                        
            except Exception as e:
                print(f"生成3D特征时出错，SMILES: {self.smile}, 错误: {str(e)}")
                if self.mol:
                    self.mol = Chem.RemoveHs(self.mol)
                self.edge_feat = None

    def _extract_3d_features(self):
        """提取3D特征：完整20维边特征"""
        try:
            conf = self.mol.GetConformer()
            n = self.mol.GetNumAtoms()
            coords = []
            for i in range(n):
                try:
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                except:
                    coords.append([0.0, 0.0, 0.0])
            coords = np.array(coords, dtype=np.float32)
            coords = np.nan_to_num(coords, nan=0.0)

            # 标量特征
            # 二面角
            for path in Chem.FindAllPathsOfLengthN(self.mol, 4, useBonds=False):
                if len(path) == 4:
                    try:
                        angle = Chem.rdMolTransforms.GetDihedralDeg(conf, path[0], path[1], path[2], path[3])
                        if not np.isnan(angle):
                            self.dihedral_angles.append(float(angle))
                    except:
                        continue

            # 原子对距离
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        distance = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                        if not np.isnan(distance):
                            self.pairwise_distances.append(float(distance))
                    except:
                        continue

            # 质心距离
            centroid = np.mean(coords, axis=0)
            for i in range(n):
                distance = float(np.linalg.norm(coords[i] - centroid))
                if not np.isnan(distance):
                    self.centroid_distances.append(distance)

            # 空间位阻
            radius = 3.0
            for i in range(n):
                neighbor_count = 0
                for j in range(n):
                    if i != j:
                        try:
                            distance = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                            if not np.isnan(distance) and distance <= radius:
                                neighbor_count += 1
                        except:
                            continue
                self.steric_hindrance.append(float(neighbor_count))

            # 分子体积
            try:
                from scipy.spatial import ConvexHull
                if n >= 4:
                    hull = ConvexHull(coords)
                    self.molecular_volume = float(hull.volume)
                else:
                    min_coords = np.min(coords, axis=0)
                    max_coords = np.max(coords, axis=0)
                    self.molecular_volume = float(np.prod(max_coords - min_coords))
            except:
                self.molecular_volume = 0.0

            # 氢键距离
            self.hbond_donor_acceptor_distances = [0.0]
            try:
                fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
                features = factory.GetFeaturesForMol(self.mol)
                donors = []
                acceptors = []
                
                for f in features:
                    if f.GetFamily() == 'Donor' and f.GetAtomIds():
                        donors.append(f.GetAtomIds()[0])
                    elif f.GetFamily() == 'Acceptor' and f.GetAtomIds():
                        acceptors.append(f.GetAtomIds()[0])
                
                self.hbond_donor_acceptor_distances = []
                for donor_idx in donors:
                    for acceptor_idx in acceptors:
                        if donor_idx != acceptor_idx and 0 <= donor_idx < n and 0 <= acceptor_idx < n:
                            try:
                                distance = Chem.rdMolTransforms.GetBondLength(conf, donor_idx, acceptor_idx)
                                if not np.isnan(distance):
                                    self.hbond_donor_acceptor_distances.append(float(distance))
                            except:
                                continue
                if not self.hbond_donor_acceptor_distances:
                    self.hbond_donor_acceptor_distances = [0.0]
            except:
                print(f"计算氢键距离时出错，SMILES: {self.smile}, 错误: {str(e)}")
                self.hbond_donor_acceptor_distances = [0.0]

            # 20边特征 - 手动填充
            edge_feat = np.zeros((n, n, 20), dtype=np.float32)
            
            # 计算距离矩阵
            dists = np.full((n, n), 5.0)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        dists[i, j] = 0.0
                    else:
                        try:
                            dists[i, j] = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                        except:
                            dists[i, j] = float(np.linalg.norm(coords[i] - coords[j]))
            
            # 每个原子的空间位阻计数
            steric_counts = np.zeros(n, dtype=np.float32)
            for i in range(n):
                steric_counts[i] = float(np.sum(dists[i] < 3.0))
            
            # 填充20维特征
            for i in range(n):
                for j in range(n):
                    dist = dists[i, j]
                    
                    # 维度0-15: 距离分箱（16个区间）
                    bins = np.linspace(0, 15, 17)
                    bin_idx = int(np.clip(np.digitize(dist, bins) - 1, 0, 15))
                    edge_feat[i, j, bin_idx] = 1.0
                    
                    # 维度16: 归一化距离
                    edge_feat[i, j, 16] = np.clip(dist / 15.0, 0.0, 1.0)
                    
                    # 维度17: 氢键潜力
                    if ((i in donors and j in acceptors) or 
                        (i in acceptors and j in donors)) and dist < 3.5:
                        edge_feat[i, j, 17] = 1.0
                    
                    # 维度18: 空间位阻差异
                    edge_feat[i, j, 18] = steric_counts[i] - steric_counts[j]
                    
                    # 维度19: 共价键存在
                    if self.mol.GetBondBetweenAtoms(i, j) is not None:
                        edge_feat[i, j, 19] = 1.0
            
            edge_feat[np.arange(n), np.arange(n), :] = 0.0
            self.edge_feat = torch.tensor(edge_feat)
            
        except Exception as e:
            traceback.print_exc()
            print(f"边特征提取异常，SMILES: {self.smile}, 错误: {str(e)}")
            n = max(10, self.mol.GetNumAtoms() if self.mol else 10)
            self.edge_feat = torch.zeros((n, n, 20))

    def task_num(self):
        return len(self.label)
    
    def change_label(self, label):
        self.label = label

    def get_3d_features(self):
        return {
            'dihedral_angles': self.dihedral_angles,
            'pairwise_distances': self.pairwise_distances,
            'centroid_distances': self.centroid_distances,
            'steric_hindrance': self.steric_hindrance,
            'molecular_volume': self.molecular_volume,
            'hbond_donor_acceptor_distances': self.hbond_donor_acceptor_distances,
            'edge_feat': self.edge_feat
        }

class MoleDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        if len(self.data) > 0:
            self.args = self.data[0].args
        else:
            self.args = None
        self.scaler = None

    def smile(self):
        smile_list = []
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list

    def mol(self):
        mol_list = []
        for one in self.data:
            mol_list.append(one.mol)
        return mol_list

    def label(self):
        label_list = []
        for one in self.data:
            label_list.append(one.label)
        return label_list

    def get_3d_features(self):
        features_3d = []
        for one in self.data:
            features_3d.append(one.get_3d_features())
        return features_3d

    def get_edge_feats(self):
        """获取边特征列表，确保失败时返回尺寸匹配的零特征"""
        edge_feats = []
        for i, one in enumerate(self.data):
            if one.edge_feat is not None:
                # 成功提取，形状为 (N_full, N_full, 20)
                #print('#',end="")
                edge_feats.append(one.edge_feat)
            else:
                # 失败情况：创建与 GAT 期望尺寸匹配的零矩阵 (N_heavy, N_heavy, 20)
                
                # 1. 尝试获取重原子数 (N_heavy)
                mol = one.mol
                if mol is None: 
                    # 极端情况：SMILES解析失败，使用一个安全值 (例如，10)
                    N_heavy = 10 
                    print('! (SMILES Fail)', end="")
                else:
                    # 获取重原子数，这是 GAT 最终需要的矩阵尺寸
                    N_heavy = mol.GetNumAtoms() 
                    print('! (3D Fail)', end="")

                # 2. 创建 (N_heavy, N_heavy, 20) 的零矩阵
                # 即使 GATEncoder 后面会进行切片匹配，这里也应使用 MolData 原始的重原子数
                # 
                # 注意：MoleData.mol 已经被 RemoveHs 了，所以 GetNumAtoms() 得到的是 N_heavy
                # 但由于您的 edge_feat 是在 AddHs 后生成的，这里的逻辑有点混乱。
                # 
                # 最安全的方法是：如果 edge_feat 为 None，我们知道 3D 生成失败了。
                # GATEncoder 期望的输入大小是 (N_heavy, N_heavy, 20)。
                
                # 让我们假设 edge_feat 成功时是 (N_full, N_full, 20)
                # 而失败时，我们必须提供 GATEncoder 可以切片（或不切片）的尺寸。
                # 考虑到 GATEncoder 最终是基于重原子数 `atom_size` 进行切片的。
                
                # 解决冲突：直接创建 GATEncoder 最终期望的尺寸 (N_heavy, N_heavy, 20)
                
                dummy_feat_heavy = torch.zeros((N_heavy, N_heavy, 20), dtype=torch.float32)
                edge_feats.append(dummy_feat_heavy)

        return edge_feats

    def task_num(self):
        if len(self.data) > 0:
            return self.data[0].task_num()
        else:
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def random_data(self, seed):
        random.seed(seed)
        random.shuffle(self.data)

    def change_label(self, label):
        assert len(self.data) == len(label)
        for i in range(len(label)):
            self.data[i].change_label(label[i])