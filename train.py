from argparse import Namespace
from logging import Logger
import numpy as np
import os
from gatgm.train import fold_train
from gatgm.tool import set_log, set_train_argument, get_task_name, mkdir


def training(args, log):
    info = log.info
    info(f'Start training. Dataset: {args.data_path}')

    first_seed = args.seed  # 第一个种子值
    data_path = args.data_path  # 数据路径
    save_path = args.save_path  # 保存路径

    scores = []

    for fold_num in range(args.num_folds):
        info(f'Seed {args.seed}')  # 打印当前的种子值
        args.seed = first_seed + fold_num  # 更新种子值为当前折数对应的种子值
        args.save_path = os.path.join(save_path, f'Seed_{args.seed}')  # 更新保存路径为当前折数对应的路径
        mkdir(args.save_path)

        fold_score = fold_train(args, log)

        scores.append(fold_score)

    scores = np.array(scores)
    info(f'Dataset: {args.data_path}')
    info(f'Running {args.num_folds} folds in total.')  # 打印总共运行的折数
    if args.num_folds > 1:
        for fold_num, fold_score in enumerate(scores):
            info(f'Seed {first_seed + fold_num} : test {args.metric} = {np.nanmean(fold_score):.6f}')  # 打印每折的测试指标值
            if args.task_num > 1:
                for task_name, task_score in zip(args.task_names, fold_score):
                    info(f'    Task {task_name} {args.metric} = {task_score:.6f}')  # 打印每个任务的测试指标值

    ave_task_score = np.nanmean(scores, axis=1)
    score_ave = np.nanmean(ave_task_score)
    score_std = np.nanstd(ave_task_score)
    info(f'Average test {args.metric} = {score_ave:.6f} +/- {score_std:.6f}')  # 打印平均测试指标值及标准差

    if args.task_num > 1:
        for i, task_name in enumerate(args.task_names):
            info(
                f'    average all-fold {task_name} {args.metric} = {np.nanmean(scores[:, i]):.6f} +/- {np.nanstd(scores[:, i]):.6f}')  # 打印每个任务的平均测试指标值及标准差

    return score_ave, score_std


if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('train', args.log_path)
    training(args, log)


