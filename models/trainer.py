import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import models
import utils









class Trainer(object):

    def __init__(self, cuda, model, optimizer,loss_fcn, scheduler,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        """

        :param cuda:
        :param model:
        :param optimizer:
                scheduler:学习率调整策略
        :param loss_fcn:
        :param train_loader:
        :param val_loader:
        :param out: 字符串，模型输出的路径，用于保存模型
        :param max_iter:
        :param size_average:
        :param interval_validate:
        """

        self.cuda = cuda

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.loss_fcn = loss_fcn

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out = out
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.viusal = utils.Visualizer()

        self.valid_loss =  0
        self.valid_acc = 0
        self.valMeanIu = 0
        self.train_loss = 0
        self.train_acc = 0
        self.trainMeanIu = 0

        self.best_mean_iu = 0

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate



    def validate(self):
        """
        用来在验证集上评估该模型，并且根据测试结果调整超参数
        :param model: 用来验证的模型
        val_loader:用来验证模型的数据集
        loss_fcn:model的损失函数
        :return:
        """
        self.model.eval()
        n_class = len(self.val_loader.dataset.class_names)
        label_trues, label_preds = [], []
        visualizations = []
        val_loss = 0
        for batch_idx, (data, target) in enumerate(self.val_loader):
            if batch_idx >50:
                break
            if self.cuda:
                data = data.cuda()
                target = target.cuda()

            print('validate' + str(batch_idx))
            with torch.no_grad():
                score = self.model(data)  # 使用模型处理输入数据得到结果

            loss = self.loss_fcn(score, target, weight=None, size_average=False)
            loss_data = loss.data.item()
            val_loss += loss / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            # 可视化模型语义分割的效果

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt =self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 15:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)

            # 计算模型在验证集的效果
        acc, acc_cls, mean_iu, fwavacc = models.label_accuracy_score(label_trues, label_preds, n_class)
        val_loss /= len(self.val_loader)
        self.scheduler.step(val_loss)

        #可视化模型的效果
        self.valid_loss = val_loss
        self.valid_acc = acc
        self.valMeanIu = mean_iu
        self.plotModelScalars()


        #保存相关的数据
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        now = datetime.datetime.now()
        utils.ModelSave(model=self.model,optim = self.optim,saveRoot=osp.join(self.out,now.strftime('%Y%m%d_%H%M%S.%f')+'checkpoint.pth.tar'),epoch= self.epoch,iteration = self.iteration)
        if mean_iu > self.best_mean_iu:
            self.best_mean_iu = mean_iu
            shutil.copy(osp.join(self.out, now.strftime('%Y%m%d_%H%M%S.%f')+'checkpoint.pth.tar'),
                        osp.join(self.out, now.strftime('%Y%m%d_%H%M%S.%f')+'model_best.pth.tar'))

        self.model.train()


    def train_epoch(self):

            if self.cuda:
                self.model = self.model.cuda()
            self.model.train()
            n_class = len(self.train_loader.dataset.class_names)

            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.cuda:
                    data = data.cuda()
                    target = target.cuda()

                print('train' + str(self.epoch) + str(batch_idx))
                iteration = batch_idx + self.epoch * len(self.train_loader)  # 将每个batch看做一次iteration,此处表示是第几个iteration
                if iteration % self.interval_validate ==0:#表示迭代训练interval_validate次后就要验证数据集，验证集的数据与训练集一致，用于评价模型的泛华能力，调整超参数
                    self.validate()

                assert self.model.training  # 判断当前是否处于训练模式中

                self.optim.zero_grad()
                score = self.model(data)
                loss = self.loss_fcn(score, target, weight=None, size_average=False)
                loss /= len(data)
                loss_data = loss.data.item()
                loss.backward()
                self.optim.step()

                # 做几次或者每次都更新统计指标并可视化,此处时每做10次可视化一下效果
                if batch_idx%10 ==0:
                    metrics = []
                    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]  # 将该像素得分最高的类看做该像素所属于的类别，所有的像素组成分类图
                    lbl_true = target.data.cpu().numpy()  # 人为标定的分类图
                    acc, acc_cls, mean_iu, fwavacc = models.label_accuracy_score(
                        lbl_true, lbl_pred, n_class=n_class)  # 这4个参数都可以作为模型在训练集的评价指标

                    metrics.append((acc, acc_cls, mean_iu, fwavacc))
                    metrics = np.mean(metrics, axis=0)

                    # 将上述标量可视化
                    self.train_loss = loss_data
                    self.iteration = iteration
                    self.train_acc = acc
                    self.TrainMeanIu = mean_iu
                    self.plotModelScalars()

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:# 如果超过了最大的迭代次数，则退出循环
                break



    #模型的效果可视化
    def plotModelScalars(self):
        """
        plotModelPer:绘制模型的相关的标量,
        将train_loss,valid_loss,train_acc,valid_acc绘制到一张图上
        :return:
        """
        # w =utils.visiualize.Visualizer()

        self.viusal.plot_scalars('modelPer', {'train_loss': self.train_loss,
                                'valid_loss': self.valid_loss,
                                'train_acc': self.train_acc,
                                'valid_acc': self.valid_acc, }, self.iteration)
        self.viusal.plot_scalars('MeanIu',{'ValMeanIu': self.valMeanIu,
                                           'TrainMeanIu':self.trainMeanIu}, self.iteration / self.interval_validate)

        self.viusal.plot_scalar('lr',self.optim.param_groups[0]['lr'],self.iteration)

    def plotModelImages(self):
        pass
