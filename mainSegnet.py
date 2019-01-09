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
import numpy as np
import torch
import torchvision
import fcn
from torch.optim import lr_scheduler
import yaml
import sys
from torch.utils.data import DataLoader
import data
import models
import torch.nn.functional as F
from distutils.version import LooseVersion
import utils
import scipy.misc



#需要重写testDataloader的代码，步骤：导入模型 准备输入数据，遍历输入数据（model(input),处理模型的输出）
def valModel(modelPth,valImagLoader,outImg,cuda=True):
    """

    :param ModelPth: 想要预测的模型的路径
    :param Img: 用于预测的输入图像
    :param OutImg: 预测后的输出图像
    :return:
    """
    #导入模型
    model = models.segnet(n_classes=5)
    utils.ModelLoad(loadRoot=modelPth,model=model)
    # import data
    testDataset = data.UAVDataClassSeg('/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/data/test.txt',
                                       train=False,test = True)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)


    model.eval()
    #读入输入图像，然后用模型去预测
     #将ValImgLoader做成loader


    for batch_idx, (datas, target) in enumerate(testLoader):
        if cuda:#是否使用GPU
            datas = datas.cuda()
            model.cuda()
            # target = target.cuda()
        with torch.no_grad():
            score = model(datas)  # 使用模型处理输入数据得到结果
        imgs = datas.data.cpu()
        img, lt = testLoader.dataset.untransform(imgs[0])
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        _ = fcn.utils.label2rgb(lbl=lbl_pred,img = img)
        scipy.misc.imsave('./t.jpg',_[0])
        # lbl_true = target.datas.cpu()
    #将预测后的结果保存为输出图像




def train():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Setup model
    model = models.segnet(n_classes=5)
    #用预训练的Vgg16网络初始化FCN32s的参数
    model.init_vgg16_params(torchvision.models.vgg16(pretrained=True))

    # Setup Dataloader,训练集和验证集数据
    """data.picFulPath('/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/train.txt',
               '/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/img/',
               '/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/cls/')
    train_dataset = data.SBDClassSeg('/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/data/ImagAndLal.txt')
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=False, drop_last=True)

    data.picFulPath('/home/mlxuan/project/DeepLearning/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
                    '/home/mlxuan/project/DeepLearning/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/',
                    '/home/mlxuan/project/DeepLearning/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/',
                    destPath='/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/data/ValImagAndLal.txt',
                    ImgFix='.jpg',lblFix='.png')

    val_dataset = data.VOCClassSeg('/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/data/ValImagAndLal.txt',train=False)
    valloader = DataLoader(val_dataset,batch_size=1,shuffle=False)"""

    train_dataset = data.UAVDataClassSeg(txt_path = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/train/trainFull.txt')
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_dataset = data.UAVDataClassSeg('/home/mlxuan/project/DeepLearning/data/image_Segmentation/valid/validFull.txt',train=False)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)




    # Setup optimizer, lr_scheduler and loss function(优化器、学习率调整策略、损失函数)

    def cross_entropy2d(input, target, weight=None, size_average=True):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):#简单的版本比较操作，此处传入的时torch.__version__,所以比较的时torch的版本
            # ==0.2.X
            log_p = F.log_softmax(input)
        else:
            # >=0.3
            log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c) log_p是对input做log_softmax后的结果，表示每个类的概率。tensor.transpose将tensor的维度交换，如行变成列，列变成行
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight)
        if size_average:
            loss /= mask.data.sum()
        return loss


    lossFun = cross_entropy2d


    optim = torch.optim.Adam(
           params=model.parameters(),
        lr=1.0e-5,
        weight_decay=0.0005)
    #定义学习率调整策略
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=1,min_lr=10e-10,eps=10e-9)  # min表示当指标不在降低时，patience表示可以容忍的step次数

    # utils.ModelLoad('/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/output/Model.path/20181227_220035.852449model_best.pth.tar',model,optim)

    trainer = models.Trainer(
        cuda =True,
        model=model,
        optimizer=optim,
        loss_fcn=lossFun,
        train_loader=trainloader,
        val_loader=valloader,
        out='./output_segnet/',
        max_iter=40000,
        scheduler = scheduler,
        interval_validate=2000
    )
    trainer.train()#进入训练




if __name__ == '__main__':
    # train()
    valModel('/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/output_segnet/20190105_174614.066962model_best.pth.tar','','',cuda=True)
