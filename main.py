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




def validate(model,valloader,loss_fcn):
    """
    用来在验证集上评估该模型，并且根据测试结果调整超参数
    :param model: 用来验证的模型
    val_loader:用来验证模型的数据集
    loss_fcn:model的损失函数
    :return:
    """
    model.eval()
    n_class = len(valloader.dataset.class_names)

    val_loss = 0
    for batch_idx,(data,target) in enumerate(valloader):
        data = data.cuda()
        target = target.cuda()

        print('validate'+str(batch_idx))
        with torch.no_grad():
            score = model(data)#使用模型处理输入数据得到结果

        loss  = loss_fcn(score,target,weight=None, size_average=False)
        loss_data = loss.data.item()
        val_loss +=loss/len(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:,:,:]
        lbl_true = target.data.cpu()

        #可视化模型语义分割的效果
        label_trues, label_preds = [], []
        visualizations = []
        for img,lt,lp in zip(imgs,lbl_true,lbl_pred):
            img,lt = valloader.dataset.untransforms(img,lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) <9:
                viz  = fcn.utils.visualize_segmentation(
                    lbl_pred = lp,lbl_true = lt,img = img,n_class = n_class)
                visualizations.append(viz)

        #计算模型在验证集的效果
    acc, acc_cls, mean_iu, fwavacc = models.label_accuracy_score(label_trues,label_preds,n_class)
    val_loss /= len(valloader)

    utils.Vis.plot_scalar('ValLos',loss_data,batch_idx)
    utils.Vis.plot_scalar('ValMeanIu',mean_iu,None)
    # utils.ModelSave(model,optim=)
    model.train()






def train_epoch(model,optim,loss_fcn,trainloader,valloader,epoch,interval_validate =4000,max_iter=40000):
    """
    训练一个epoch
    :param model: 用于训练的模型
            optim:训练时所采用的优化器
            loss_fcn:训练时采用的损失函数
           trainloader:用于训练的数据集
           valloader:用于验证的数据集
           epoch:表示这是第几个epoch
    :return:
    """
    model = model.cuda()



    model.train()
    n_class = len(valloader.dataset.class_names)
    for batch_idx,(data,target) in enumerate(trainloader):
        data = data.cuda()
        target = target.cuda()

        print('train' +str(epoch)+str(batch_idx))
        iteration = batch_idx + epoch * len(trainloader)  #将每个batch看做一次iteration,此处表示是第几个iteration
        # if iteration % interval_validate ==400:#表示迭代训练interval_validate次后就要验证数据集，验证集的数据与训练集一致，用于评价模型的泛华能力，调整超参数
        #     validate(model=model,valloader=valloader,loss_fcn=loss_fcn)

        assert model.training #判断当前是否处于训练模式中

        optim.zero_grad()
        score = model(data)
        loss  = loss_fcn(score,target,weight=None, size_average=False)
        loss /=len(data)
        loss_data = loss.data.item()
        loss.backward()
        optim.step()

        #做几次或者每次都更新统计指标并可视化
        metrics = []
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:,:,:]#将该像素得分最高的类看做该像素所属于的类别，所有的像素组成分类图
        lbl_true = target.data.cpu().numpy()#人为标定的分类图
        acc,acc_cls,mean_iu,fwavacc = models.label_accuracy_score(
            lbl_true,lbl_pred,n_class=n_class)#这4个参数都可以作为模型在训练集的评价指标

        metrics.append((acc,acc_cls,mean_iu,fwavacc))
        metrics = np.mean(metrics,axis = 0)

        #将上述标量可视化
        utils.Vis.plot_scalar('loss2',loss_data,iteration)
        if iteration > max_iter:#如果超过了最大的迭代次数，则退出循环
            break


















def train():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Setup model
    model = models.FCN32s()
    #用预训练的Vgg16网络初始化FCN32s的参数
    vgg16 = models.VGG16(pretrained=True)
    model.copy_params_from_vgg16(vgg16)

    # Setup Dataloader,训练集和验证集数据
    data.picFulPath('/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/train.txt',
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
    valloader = DataLoader(val_dataset,batch_size=1,shuffle=False)


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

    def get_parameters(model, bias=False):
        import torch.nn as nn
        modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            nn.Sequential,
            models.FCN32s,


        )
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                # weight is frozen because it is just a bilinear upsampling
                if bias:
                    assert m.bias is None
            elif isinstance(m, modules_skipped):
                continue
            else:
                raise ValueError('Unexpected module: %s' % str(m))


    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr':  1.0e-5* 2, 'weight_decay': 0},
        ],
        lr=1.0e-5,
        momentum=0.99,
        weight_decay=0.0005)
    #定义学习率调整策略
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=0,min_lr=10e-10,eps=10e-8)  # min表示当指标不在降低时，patience表示可以容忍的step次数

    utils.ModelLoad('./output/Model.path',model,optim)

    trainer = models.Trainer(
        cuda =True,
        model=model,
        optimizer=optim,
        loss_fcn=lossFun,
        train_loader=trainloader,
        val_loader=valloader,
        out='./output/Model.path',
        max_iter=40000,
        scheduler = scheduler
    )
    trainer.train()#进入训练




if __name__ == '__main__':
    train()

