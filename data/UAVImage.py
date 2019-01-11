

import os
import glob
import random
import shutil


import collections
import os.path as osp

import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
import fcn

class UAVDataClassSeg(data.Dataset):
    """
    class_names = np.array([
        'background',
        'vegetation',
        'building',
        'water',
        'road' ])
        """
    class_names = np.array([
        'background',
        'Road',
        'Tree',
        'Grass',
        'GrassAndSoil',
        'Soil',
        'withered grass',
        'Water',
        'Building',
        'GreenHouse'
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])


    def __init__(self, txt_path, transforms=None, train=True,test = False):#当train=True时，表示是训练集；ttrain=False且test=False,表示是验证集，test=False而test=True时，是测试集
        '''
         参数：txt_path包含了每张图片的路径和图片的标签的路径（语义分割时就是标记好的图片的路径）
             transforms表示该读取图片后，图片要进行的预处理，如果为None,则使用默认的预处理
         '''

        # 1.指定数据集，常用的方法是给定图片路径的txt文档,将他们读入一个列表
        imgs = []

        self.train = train  # 表示此次时训练集还是数据集，可能需要用来做不同的数据预处理
        self.test = test

        fh = open(txt_path, 'r')
        # 遍历整个txt文档，每张图片的路径作为list的一个元素
        for line in fh:
            line = line.rstrip()  # 去掉改行最后的回车符号
            words = line.split()
            imgs.append((words[0], words[1])if len(words)==2 else(words[0],))  # words[0]是原图片的路径，words[1]是标记好的图片的标签
        self.imgs = imgs

        # 2.指定读入图片后的预处理
        if transforms is None:

            def transforms(img, lbl = None):
                # T.Resize T.CenterCrop将图片保持纵横比缩放裁剪为同一大小，此处图片已经预先裁剪为指定的大小，所以不需要裁剪
                # T.Tensor,T.Normalize 将图片转为[0,1]的Tensor,并归一化
                if self.train == True:#训练集
                    transImg = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([T.ToPILImage(),
                                          T.ToTensor()])
                    img = transImg(img)
                    lbl = transLbl(lbl).long()
                elif self.test == False:#验证集
                    transImg = T.Compose([T.ToTensor(),
                                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([T.ToPILImage(), T.ToTensor()])
                    img = transImg(img)
                    lbl = transLbl(lbl).long()
                else: #测试集
                    transImg = T.Compose([T.ToTensor(),
                                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    img = transImg(img)
                    lbl = 0
                return img, lbl  # lbl的训练集和验证集都很定时tensor,此处将他们都设计为3维的tensor[1*H*W]

            def untransforms(img, lbl = None):
                # 该trans是对已经归一化的tensor处理，得到归一化之前的Tensor
                trans = T.Compose(
                    [T.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])])
                imgUntrans = trans(img)
                # for i in range(img.shape[0]):
                #     imgUntrans[i] = trans(img[i])

                return ((255 * imgUntrans).type(torch.ByteTensor).transpose(0, 1).transpose(1, 2)).numpy(), \
                     None if lbl is None else (lbl.squeeze(0)).numpy()

            # 常规的数据操作：（裁剪为统一大小,可选T.scale），（数据增强，如随机裁剪等 语义分割时一般不做这个），ToTensor()后+T.Normalize
            # if self.train:  # 如果此次是训练集（训练集和验证集可能读取数据方法一样，但是预处理的过程不一样）
        self.transforms = transforms
        self.untransform = untransforms
        fh.close()



    def __getitem__(self, index):  # 该方法是给定索引或键值，返回对应的值，常用在enumerate遍历数据集时
        '''返回一张图片的验证集和测试集的数据'''
        # 从list中获取图片的路径
        img_path = self.imgs[index][0]
        lbl_path = self.imgs[index][1] if len(self.imgs[index])==2 else None
        # 读取图片，为numpy格式
        im = Image.open(img_path)
        # lbl是mat格式，mat格式的处理参考https://blog.csdn.net/qq_32425195/article/details/85202552

        # load label
        if self.test !=True and lbl_path != None:
            lbl = Image.open(lbl_path)
            lbl = np.array(lbl, dtype=np.int32)
            lbl[lbl == 255] = -1
            lbl = (torch.from_numpy(lbl)).unsqueeze(0)
        else:
            lbl = None
        return self.transforms(im, lbl)


    def __len__(self):
        '''
        返回为数据集中所有图片的个数
        :return:
        '''
        return len(self.imgs)
if __name__ == '__main__':
    """
    测试数据集类是否正确，
    初始化
    遍历
    测试其他函数(untransform)
    """
    """
    visualizations = []
    trainDataset = UAVDataClassSeg(txt_path = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/train/trainFull.txt')
    trainloader = DataLoader(trainDataset, batch_size=8, shuffle=True, drop_last=True)
    for batch_idx, (data, target) in enumerate(trainloader):
        for img, lt in zip(data, target):
            img, lt = trainDataset.untransform(img, lt)
            if len(visualizations) < 15:
                 viz = fcn.utils.visualize_segmentation(lbl_pred=lt, lbl_true=lt, img=img, n_class=5)
                 visualizations.append(viz)

    scipy.misc.imsave('./visualization_viz.jpg', fcn.utils.get_tile_image(visualizations))
"""
    """
    测试test集的数据是否可以
    """
    trainDataset = UAVDataClassSeg(txt_path = './test.txt',train=False,test=True)
    trainloader = DataLoader(trainDataset, batch_size=1, shuffle=True, drop_last=True)
    for batch_idx, (data, target) in enumerate(trainloader):
        img, lt = trainDataset.untransform(data[0])#data中包含的图片的数目由Dataloader的batch_size决定，data[0]表示第一张图片
        Image.fromarray(img).save('./1.jpg', 'JPEG')