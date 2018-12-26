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

    def __init__(self, cuda, model, optimizer,loss_fcn,
                 train_loader, val_loader, out, max_iter,
                 size_average=False, interval_validate=None):
        """

        :param cuda:
        :param model:
        :param optimizer:
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
        self.loss_fcn = loss_fcn

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.out = out
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
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

        val_loss = 0
        for batch_idx, (data, target) in enumerate(self.val_loader):
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
            label_trues, label_preds = [], []
            visualizations = []
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt =self.val_loader.dataset.untransforms(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)

            # 计算模型在验证集的效果
        acc, acc_cls, mean_iu, fwavacc = models.label_accuracy_score(label_trues, label_preds, n_class)
        val_loss /= len(self.val_loader)

        utils.Vis.plot_scalar('ValLos', loss_data, batch_idx)
        utils.Vis.plot_scalar('ValMeanIu', mean_iu, None)
        utils.ModelSave(model=self.model,optim = self.optim,saveRoot=self.out,epoch= self.epoch,iteration = self.iteration)
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

                # 做几次或者每次都更新统计指标并可视化
                metrics = []
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]  # 将该像素得分最高的类看做该像素所属于的类别，所有的像素组成分类图
                lbl_true = target.data.cpu().numpy()  # 人为标定的分类图
                acc, acc_cls, mean_iu, fwavacc = models.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)  # 这4个参数都可以作为模型在训练集的评价指标

                metrics.append((acc, acc_cls, mean_iu, fwavacc))
                metrics = np.mean(metrics, axis=0)

                # 将上述标量可视化
                utils.Vis.plot_scalar('loss2', loss_data, iteration)


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:# 如果超过了最大的迭代次数，则退出循环
                break
