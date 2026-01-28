from argparse import Namespace
from logging import Logger
import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR
from gatgm.tool.tool import load_model
from gatgm.model import FPGNN
from gatgm.data import MoleDataSet
from torch.utils.data import DataLoader

def epoch_train(model, data, loss_f, optimizer, scheduler, args):
    model.train()
    data.random_data(args.seed)
    loss_sum = 0
    data_used = 0
    
    # num_workers > 0 启用多进程数据加载，释放主进程/GPU等待时间
    # MoleDataSet 已经实现了 __len__ 和 __getitem__，所以可以直接用
    # DataLoader会自动调用 __getitem__ 方法
    train_loader = DataLoader(dataset=data, 
                              batch_size=args.batch_size, 
                              shuffle=False, # 已经在 data.random_data() 中处理
                              num_workers=4, # 根据您的CPU核心数调整
                              collate_fn=lambda x: MoleDataSet(x)) # 使用您的 MoleDataSet 作为批处理函数

    for batch_data in train_loader: # 迭代 DataLoader
        data_now = batch_data # data_now 现在是一个 MoleDataSet 实例
        smile = data_now.smile()
        label = data_now.label()
        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])
        
        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()
        
        weight = torch.ones(target.shape)
        if args.cuda:
            weight = weight.cuda()
        
        model.zero_grad()
        
        # 支持3D边特征
        if hasattr(args, 'use_3d_features') and args.use_3d_features:
            # edge_feats 是一个列表，列表中的每个元素是 (N,N,20) 的 Tensor
            edge_feats = data_now.get_edge_feats() 
            # batch size > 1 时需要一个列表，batch size == 1 时需要 [tensor]
            pred = model(smile, edge_feat=edge_feats)
        else:
            pred = model(smile)
        
        loss = loss_f(pred, target) * weight * mask
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        data_used += len(smile) # 确保 data_used 的累加正确
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    
    if isinstance(scheduler, ExponentialLR):
        scheduler.step()
    return loss_sum / data_used if data_used > 0 else 0

def predict(model, data, batch_size, scaler):
    model.eval()
    pred = []
    data_total = len(data)
    
    for i in range(0, data_total, batch_size):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()
        
        with torch.no_grad():
            if hasattr(model.args, 'use_3d_features') and model.args.use_3d_features:
                edge_feats = data_now.get_edge_feats()
                if len(edge_feats) == len(smile):
                    batch_pred = model(smile, edge_feat=edge_feats[0] if len(edge_feats) == 1 else edge_feats)
                else:
                    batch_pred = model(smile)
            else:
                batch_pred = model(smile)
        
        pred.append(batch_pred.cpu().numpy())
    
    return np.vstack(pred)

def compute_score(pred, label, metric_f, args, log):
    info = log.info
    batch_size = args.batch_size
    task_num = args.task_num
    data_type = args.dataset_type
    
    if len(pred) == 0:
        return [float('nan')] * task_num
    
    pred_val = []
    label_val = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)
    
    result = []
    for i in range(task_num):
        if data_type == 'classification':
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue
        re = metric_f(label_val[i], pred_val[i])
        result.append(re)
    
    return result

def fold_train(args, log):
    info = log.info
    debug = log.debug
    
    debug('Start loading data')
    
    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path, args)
    args.task_num = data.task_num()
    data_type = args.dataset_type
    if args.task_num > 1:
        args.is_multitask = 1
    
    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path, args)
    if args.test_path:
        test_data = load_data(args.test_path, args)
    if args.val_path and args.test_path:
        train_data = data
    elif args.val_path:
        split_ratio = (args.split_ratio[0], 0, args.split_ratio[2])
        train_data, _, test_data = split_data(data, args.split_type, split_ratio, args.seed, log)
    elif args.test_path:
        split_ratio = (args.split_ratio[0], args.split_ratio[1], 0)
        train_data, val_data, _ = split_data(data, args.split_type, split_ratio, args.seed, log)
    else:
        train_data, val_data, test_data = split_data(data, args.split_type, args.split_ratio, args.seed, log)
    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}    Test size: {len(test_data)}')
    
    if data_type == 'regression':
        label_scaler = get_label_scaler(train_data)
    else:
        label_scaler = None
    args.train_data_size = len(train_data)
    
    loss_f = get_loss(data_type)
    metric_f = get_metric(args.metric)
    
    debug('Training Model')
    model = FPGNN(args)
    debug(model)
    if args.cuda:
        model = model.to(torch.device("cuda"))
    save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[args.warmup_epochs], total_epochs=[args.epochs] * args.num_lrs,
                       steps_per_epoch=args.train_data_size // args.batch_size, init_lr=[args.init_lr], max_lr=[args.max_lr],
                       final_lr=[args.final_lr])
    if data_type == 'classification':
        best_score = -float('inf')
    else:
        best_score = float('inf')
    best_epoch = 0

    no_improvement_count = 0
    
    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        
        train_loss = epoch_train(model, train_data, loss_f, optimizer, scheduler, args)
        
        train_pred = predict(model, train_data, args.batch_size, label_scaler)
        train_label = train_data.label()
        train_score = compute_score(train_pred, train_label, metric_f, args, log)
        val_pred = predict(model, val_data, args.batch_size, label_scaler)
        val_label = val_data.label()
        val_score = compute_score(val_pred, val_label, metric_f, args, log)

        info(f'Train loss = {train_loss:.6f}')
        
        ave_train_score = np.nanmean(train_score)
        print()
        info(f'Train {args.metric} = {ave_train_score:.6f}')
        
        ave_val_score = np.nanmean(val_score)
        info(f'Validation {args.metric} = {ave_val_score:.6f}')
        if args.task_num > 1:
            for one_name, one_score in zip(args.task_names, val_score):
                info(f'Validation {one_name} {args.metric} = {one_score:.6f}')
        
        improved = False
        if data_type == 'classification' and ave_val_score > best_score:
            improved = True
        elif data_type == 'regression' and ave_val_score < best_score:
            improved = True

        if improved:
            best_score = ave_val_score
            best_epoch = epoch
            no_improvement_count = 0 # 重置计数器
            save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)
        else:
            no_improvement_count += 1
            info(f'No improvement for {no_improvement_count}/{args.patience} epochs.')

        if no_improvement_count >= args.patience:
            info(f'Early stopping at epoch {epoch} as validation {args.metric} did not improve for {args.patience} consecutive epochs.')
            break # 退出训练循环
    
    info(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
    
    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log)
    test_smile = test_data.smile()
    test_label = test_data.label()
    
    test_pred = predict(model, test_data, args.batch_size, label_scaler)
    test_score = compute_score(test_pred, test_label, metric_f, args, log)
    
    ave_test_score = np.nanmean(test_score)
    info(f'Seed {args.seed} : test {args.metric} = {ave_test_score:.6f}')
    
    if args.task_num > 1:
        for one_name, one_score in zip(args.task_names, test_score):
            info(f'Task {one_name} {args.metric} = {one_score:.6f}')
    

    return test_score
