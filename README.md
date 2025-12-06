# GAT-GM
<font size=1>Prediction of Drug Properites using GAT-FPGNN with 3D Features</font>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C?logo=pytorch&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-11.7-success?logo=nvidia&logoColor=white)](#)
[![RDKit](https://img.shields.io/badge/RDKit-2024.03.6-00CC00?logo=rdkit&logoColor=white)](#)
[![Model](https://img.shields.io/badge/Model-FPGNN%20%2B%203D--GAT-ff69b4)](#)
[![3D Edge Features](https://img.shields.io/badge/3D%20Edge%20Features-20dim-brightgreen)](#)
[![Mixed FP](https://img.shields.io/badge/Fingerprint-Mixed%201489dim-ff9900)](#)
[![Tasks](https://img.shields.io/badge/Tasks-Classification%20%26%20Regression-blueviolet)](#)
[![Split](https://img.shields.io/badge/Split-Random%20%7C%20Scaffold-00bcd4)](#)
---
# 使用 (Usage)
## 1.环境部署：
- 本研究基于Ubuntu系统，强烈建议使用Ubuntu系统运行
- 构建环境使用<a href="https://anaconda.org/">anaconda</a>，点击前往官网下载，可以参考<a href="https://blog.csdn.net/JineD/article/details/129507719">这篇教程</a>
<br>
使用environment文件下载，按照自己是否有NVIDIA显卡选择cpu/cuda

```
$ conda env create -f environment-cpu.txt
```  
```
$ conda env create -f environment-cuda.txt
```  
之后请检查numpy的版本，确保在1.x（建议1.26.4）  

将FPGNN项目包放在与gatgm包的同级文件  
FPGNN的下载地址https://github.com/idrugLab/FP-GNN  

文件结构：  
```
├── data  *训练用数据集
├── fpgnn  
├── gatgm  
│   ├── data  
│   │   ├── data.py  
│   │   ├── graph.py  
│   │   ├── __init__.py  
│   │   ├── pubchemfp.py  
│   │   └── scaffold.py  
│   ├── __init__.py  
│   ├── model  
│   │   ├── gatgm.py  
│   │   └──__init__.py  
│   ├── tool  
│   │   ├── abstract.py  
│   │   ├── args.py  
│   │   ├── __init__.py  
│   │   └── tool.py  
│   └── train  
│       ├── __init__.py  
│       └── train.py  
├── hyper_opti.py  
├── predict.py  *预测
├── train.py  *训练
└── utils.py  
```

## 2.运行
### 通用参数
| 参数名                  | 类型/可选值                          | 默认值            | 说明                                                                 |
|------------------------|-------------------------------------|------------------|----------------------------------------------------------------------|
| `--data_path`          | str                                 | 无（必填）        | 输入的训练数据集 CSV 文件路径                                          |
| `--save_path`          | str                                 | `model_save`     | 训练好的模型保存目录                                                  |
| `--log_path`           | str                                 | `log`            | 日志文件保存目录                                                      |
| `--dataset_type`       | `classification` / `regression`     | 无（必填）        | 数据集任务类型：分类或回归                                             |
| `--is_multitask`       | int (0/1)                           | 0                | 是否为多任务学习，0=单任务，1=多任务                                    |
| `--task_num`           | int                                 | 1                | 多任务学习中的任务数量                                                 |
| `--seed`               | int                                 | 0                | 随机种子（用于数据划分和模型初始化）                                    |

### 数据划分相关

| 参数名                  | 类型/可选值                          | 默认值                  | 说明                                                                 |
|------------------------|-------------------------------------|------------------------|----------------------------------------------------------------------|
| `--split_type`         | `random` / `scaffold`               | `random`               | 数据划分方式：随机划分或基于骨架（scaffold）划分                        |
| `--split_ratio`        | float（三个数）                      | `[0.8, 0.1, 0.1]`      | 训练/验证/测试集划分比例，例如 0.8 0.1 0.1                            |
| `--val_path`           | str                                 | None                   | 额外的验证集 CSV 文件路径（可选）                                      |
| `--test_path`          | str                                 | None                   | 额外的测试集 CSV 文件路径（可选）                                      |
| `--num_folds`          | int                                 | 1                      | 交叉验证的折数（>1 时启用k折交叉验证）                                  |

### 训练超参数

| 参数名                  | 类型/可选值                          | 默认值            | 说明                                                                 |
|------------------------|-------------------------------------|------------------|----------------------------------------------------------------------|
| `--epochs`             | int                                 | 30               | 训练总轮数                                                           |
| `--patience`           | int                                 | 10               | 早停耐心值：验证集指标多少轮不提升就停止训练                            |
| `--batch_size`         | int                                 | 50               | 批次大小                                                             |
| `--metric`             | `auc` / `prc-auc` / `rmse`          | 自动根据任务选择   | 评估指标，分类默认 auc，回归默认 rmse                                 |

### 指纹与特征设置

| 参数名                  | 类型/可选值                          | 默认值            | 说明                                                                 |
|------------------------|-------------------------------------|------------------|----------------------------------------------------------------------|
| `--fp_type`            | `morgan` / `mixed`                  | `mixed`          | 分子指纹类型：纯Morgan指纹 或 混合指纹                                |
| `--use_3d_features`    | 标志（无需值）                       | False            | 是否启用3D空间特征（用于GAT）                                         |
| `--edge_dim`           | int                                 | 20               | 3D图中边特征的维度（仅在使用3D特征时有效）                             |
| `--gat_scale`          | float                               | 0.5              | GAT分支与FPN分支融合时的权重缩放因子                                   |

### GAT（图注意力网络）参数

| 参数名                  | 类型                  | 默认值     | 说明                                      |
|------------------------|-----------------------|-----------|-------------------------------------------|
| `--nhid`               | int                   | 64        | 每个注意力头的隐藏层维度                   |
| `--nheads`             | int                   | 4         | 注意力头数                                |
| `--dropout_gat`        | float                 | 0.1       | GAT层的Dropout比例                        |

## FPN（指纹全连接网络）参数

| 参数名                  | 类型                  | 默认值     | 说明                                      |
|------------------------|-----------------------|-----------|-------------------------------------------|
| `--fp_2_dim`           | int                   | 256       | 指纹分支第一层全连接后的维度               |
| `--dropout`            | float                 | 0.1       | 指纹分支的Dropout比例                     |

### 共享模型维度

| 参数名                  | 类型                  | 默认值     | 说明                                      |
|------------------------|-----------------------|-----------|-------------------------------------------|
| `--hidden_size`        | int                   | 256       | GAT、FPN、最终FFN共享的隐藏层维度          |

### 预测模式专用参数（predict / interfp / intergraph）

| 参数名                  | 类型                  | 默认值            | 说明                                      |
|------------------------|-----------------------|------------------|-------------------------------------------|
| `--predict_path`       | str                   | 无（必填）        | 待预测的分子 CSV 文件路径                  |
| `--model_path`         | str                   | 无（必填）        | 训练好的模型文件路径（.pt）                |
| `--result_path`        | str                   | `result.txt`     | 预测结果输出路径（predict模式）            |
| `--figure_path`        | str                   | `figure`         | 可视化图表保存目录（intergraph模式）       |

### 超参数搜索专用参数

| 参数名                  | 类型                  | 默认值     | 说明                                      |
|------------------------|-----------------------|-----------|-------------------------------------------|
| `--search_num`         | int                   | 10        | 超参数搜索的试验次数（用于hyper模式）      |

### 自动设置（无需手动指定）

| 参数名                  | 值                                | 说明                                      |
|------------------------|-----------------------------------|-------------------------------------------|
| `--cuda`               | 自动检测是否可用GPU                | True/False                                |
| `--init_lr`            | 1e-4                              | 初始学习率                                 |
| `--max_lr`             | 1e-3                              | OneCycleLR最大学习率                       |
| `--final_lr`           | 1e-4                              | 最终学习率                                 |
| `--warmup_epochs`      | 2.0                               | 学习率预热轮数                             |

### 使用模式对应函数

| 运行模式         | 对应函数                     | 主要用途                     |
|------------------|-----------------------------|-----------------------------|
| 训练             | `set_train_argument()`      | 正常训练模型                 |
| 预测             | `set_predict_argument()`    | 使用训练好的模型进行预测      |
| 超参数搜索       | `set_hyper_argument()`      | 进行超参数自动搜索           |
| 指纹解释性分析   | `set_interfp_argument()`    | 分析重要指纹位               |
| 图结构解释性分析 | `set_intergraph_argument()` | 生成注意力图可视化            |


### e.g.
训练
```
$ python train.py --data_path data/bace.csv --dataset_type regression --use_3d_features --gat_scale 0.5 --hidden_size 300 --fp_2_dim 600 --dropout 0.2 --batch_size 1024 --epochs 200 --seed 42 --save_path model_save/bace --patience 30
```

预测
```
$ python predict.py --predict_path your_dataset.csv --mo
del_path model_save/bace/Seed_42/model.pt --result_path result/your_result.csv --batch_size 512
```

**--bach_size**根据显存调整，**--dataset_type**根据任务类型（分类/回归）调整，**--seed**随机种子不同可能有不同的效果

<br>

#  原理 (Principle)

本项目实现了一个**指纹 + 3D增强图注意力**双分支融合模型，将分子3D空间几何信息以边特征形式注入GAT注意力机制。
---

## 整体架构

<img width="870" height="312" alt="image" src="https://github.com/user-attachments/assets/79bb1b65-541b-44d8-ba46-48b96d999b0c" />

融合比例由 `--gat_scale`（0~1）控制：1=纯GAT，0=纯指纹，0.5=等权融合。

## 20维3D边特征增强的图注意力

传统GAT仅依赖拓扑结构，本项目在注意力分数中显式加入3D几何信息：

$$e_{ij} = \text{LeakyReLU}\Big( \mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_j] + \mathbf{v}^\top \mathbf{e}_{ij}^{3D} \Big)$$

$\mathbf{e}_{ij}^{3D} \in \mathbb{R}^{20}$ 包含距离分箱、氢键潜力、位阻差等。即使3D生成失败也自动补零。

## 指纹分支（FPN）

- `morgan` → 1024维Morgan指纹  
- `mixed`（默认）→ MACCS(166) + ErG(307) + PubChem(881) = 1489维

经两层全连接映射到隐藏维度 256。

## 特征融合与预测头

设 $\lambda = $ `--gat_scale`，融合方式为：

$$
\mathbf{h}_{\text{fusion}} =
\begin{cases}
\mathbf{h}_{\text{GAT}} & \lambda=1 \\
\mathbf{h}_{\text{FPN}} & \lambda=0 \\
\text{Concat}(\mathbf{W}_1\mathbf{h}_{\text{GAT}}, \mathbf{W}_2\mathbf{h}_{\text{FPN}}) & 0<\lambda<1
\end{cases}
$$

最终预测头（2层隐藏层 + Dropout）：

$$\text{FFN}:\ \mathbb{R}^{512}\to\mathbb{R}^{256}\to\mathbb{R}^{\text{task-num}}$$

- 分类任务：训练用 BCEWithLogitsLoss，推理自动加 Sigmoid  
- 回归任务：直接 MSE

## 训练策略

- 优化器：Adam + Noam学习率调度（2 epoch warmup → 峰值1e-3 → 衰减至1e-4）
- 早停：验证集连续 `--patience=10` 轮不提升即停止
- 支持多任务学习、随机/scaffold划分、k折交叉验证

通过精准的3D空间信息与高效指纹表征的互补，本模型在多种分子属性预测任务上显著优于单一分支模型。
<br>
# 注释
1.该研究参考了<a href="https://github.com/idrugLab/FP-GNN">FP-GNN</a>模型  
2.该模型由大连医科大学附一临床2023**刘骐**、基础生工2021**马铭泽**、基础生工2019**张斌**共同完成，由大连医科大学基础医学院**宋涛**老师指导，使用请注明出处  
3.该模型通常情况下回归任务比分类任务收敛的更快，但效果没有分类任务好  
4.建议在运行过程中使用Htop工具监视CPU状态，下载：
```
$ sudo apt install htop
```
同时监视显卡状态：
```
$ watch -n 1 nvidia-smi
```
5.所有模型都不是万能的，该模型在某些数据集的训练效果极佳：bbbp seed=42 Average test auc = 0.940869 +/- 0.000000  
但部分数据集表现较差，部分情况下切换种子会更有效，否则建议使用其他模型
<br><br>&emsp;
