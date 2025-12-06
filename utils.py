import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score

def mkdir(path, isdir=True):
    if isdir == False:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def set_log(name, save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)
    mkdir(save_path)
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    return log

def get_header(path):
    with open(path) as file:
        header = next(csv.reader(file))
    return header

def get_task_name(path):
    task_name = get_header(path)[1:]
    return task_name

def load_data(path, args):
    from gatgm.data import MoleDataSet, MoleData  # 延迟导入
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        lines = []
        for line in reader:
            lines.append(line)
        data = []
        for line in lines:
            one = MoleData(line, args)
            data.append(one)
        data = MoleDataSet(data)
        fir_data_len = len(data)
        data_val = []
        smi_exist = []
        for i in range(fir_data_len):
            if data[i].mol is not None:
                smi_exist.append(i)
        data_val = MoleDataSet([data[i] for i in smi_exist])
        now_data_len = len(data_val)
        print('There are ', now_data_len, ' smiles in total.')
        if fir_data_len - now_data_len > 0:
            print('There are ', fir_data_len, ' smiles first, but ', fir_data_len - now_data_len, ' smiles is invalid.')
        return data_val

def split_data(data, type, size, seed, log):
    from gatgm.data import MoleDataSet, scaffold_split  # 延迟导入
    assert len(size) == 3
    assert sum(size) == 1
    if type == 'random':
        data.random_data(seed)
        train_size = int(size[0] * len(data))
        val_size = int(size[1] * len(data))
        train_val_size = train_size + val_size
        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]
        return MoleDataSet(train_data), MoleDataSet(val_data), MoleDataSet(test_data)
    elif type == 'scaffold':
        return scaffold_split(data, size, seed, log)
    else:
        raise ValueError('Split_type is Error.')

def get_label_scaler(data):
    smile = data.smile()
    label = data.label()
    label = np.array(label).astype(float)
    ave = np.nanmean(label, axis=0)
    ave = np.where(np.isnan(ave), np.zeros(ave.shape), ave)
    std = np.nanstd(label, axis=0)
    std = np.where(np.isnan(std), np.ones(std.shape), std)
    std = np.where(std == 0, np.ones(std.shape), std)
    change_1 = (label - ave) / std
    label_changed = np.where(np.isnan(change_1), None, change_1)
    label_changed.tolist()
    data.change_label(label_changed)
    return [ave, std]

def get_loss(type):
    if type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    elif type == 'regression':
        return nn.MSELoss(reduction='none')
    else:
        raise ValueError('Data type Error.')

def prc_auc(label, pred):
    prec, recall, _ = precision_recall_curve(label, pred)
    result = auc(recall, prec)
    return result

def rmse(label, pred):
    result = mean_squared_error(label, pred)
    return math.sqrt(result)

def get_metric(metric):
    if metric == 'auc':
        return roc_auc_score
    elif metric == 'prc-auc':
        return prc_auc
    elif metric == 'rmse':
        return rmse
    else:
        raise ValueError('Metric Error.')

def save_model(path, model, scaler, args):
    if scaler != None:
        state = {
            'args': args,
            'state_dict': model.state_dict(),
            'data_scaler': {
                'means': scaler[0],
                'stds': scaler[1]
            }
        }
    else:
        state = {
            'args': args,
            'state_dict': model.state_dict(),
            'data_scaler': None
        }
    torch.save(state, path)

class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                 init_lr, max_lr, final_lr):
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)
        self.num_lrs = len(optimizer.param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)
        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return list(self.lr)

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:
                self.lr[i] = self.final_lr[i]
            self.optimizer.param_groups[i]['lr'] = self.lr[i]