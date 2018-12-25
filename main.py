"""
main：
    train:负责训练 获得模型->获得数据->确定损失函数、参数更新优化器
                   ->模型评估方法->遍历数据集训练(正向传播 损失计算 反向传播 参数优化 统计指标更新和可视化)
                   ->保存模型->在验证集上验证->超参数更新(主要是学习率)

"""

import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import torch
import yaml
import sys
from torch.utils.data import DataLoader
import data
import models


def train():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Setup model
    model = models.FCN32s()


    # Setup Dataloader
    data.picFulPath('/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/train.txt',
               '/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/img/',
               '/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/cls/')
    train_dataset = data.SBDClassSeg('/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/data/ImagAndLal.txt')
    trainloader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=True)

    # Setup optimizer, lr_scheduler and loss function
