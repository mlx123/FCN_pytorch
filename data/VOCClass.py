'''定义自己的数据集'''
# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import scipy.io
import torch
import utils
from .SBDClassSeg import picFulPath



class VOCClassSeg(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])


    def __init__(self, txt_path, transforms=None,train=True):  # 此处train val test（指定此次的数据时用来做什么的，不同用处的数据集图片预处理方法不同，可以传入不同的txt_path即可）的指定应该用union

        '''

        参数：txt_path包含了每张图片的路径和图片的标签的路径（语义分割时就是标记好的图片的路径）
            transforms表示该读取图片后，图片要进行的预处理，如果为None,则使用默认的预处理
        '''

        # 1.指定数据集，常用的方法是给定图片路径的txt文档,将他们读入一个列表
        imgs = []
        self.train = train
        fh = open(txt_path, 'r')
        # 遍历整个txt文档，每张图片的路径作为list的一个元素
        for line in fh:
            line = line.rstrip()  # 去掉改行最后的回车符号
            words = line.split()
            imgs.append((words[0], words[1]))  # words[0]是原图片的路径，words[1]是标记好的图片的标签
        self.imgs = imgs
        # 2.指定读入图片后的预处理
        if transforms is None:

            def transforms(img, lbl):
                # T.Resize T.CenterCrop将图片保持纵横比缩放裁剪为同一大小，
                # T.Tensor,T.Normalize 将图片转为[0,1]的Tensor,并归一化
                if train==True:
                    transImg = T.Compose([T.Resize(224), T.CenterCrop(224),
                                   T.ToTensor(),
                                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([T.ToPILImage(),T.Resize(224), T.CenterCrop(224),
                                       T.ToTensor()])
                else:
                    transImg = T.Compose([T.ToTensor(),
                                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([ T.ToPILImage(),T.ToTensor()])

                img = transImg(img)

                lbl = transLbl(lbl)
                return img, lbl.long()  # lbl的训练集和验证集都很定时tensor,此处将他们都设计为3维的tensor[1*H*W]

            def untransforms(img, lbl):
                # 该trans是对已经归一化的tensor处理，得到归一化之前的Tensor
                trans = T.Compose(
                    [T.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])])
                imgUntrans = trans(img)
                # for i in range(img.shape[0]):
                #     imgUntrans[i] = trans(img[i])

                return ((255*imgUntrans).type(torch.ByteTensor).transpose(0,1).transpose(1,2)).numpy(), (lbl.squeeze(0)).numpy()

            # 常规的数据操作：（裁剪为统一大小,可选T.scale），（数据增强，如随机裁剪等 语义分割时一般不做这个），ToTensor()后+T.Normalize
            #if self.train:  # 如果此次是训练集（训练集和验证集可能读取数据方法一样，但是预处理的过程不一样）
        self.transforms = transforms
        self.untransforms = untransforms
        fh.close()

    def __getitem__(self, index):  # 该方法是给定索引或键值，返回对应的值，常用在enumerate遍历数据集时
        '''返回一张图片的验证集和测试集的数据'''
        # 从list中获取图片的路径
        img_path = self.imgs[index][0]
        lbl_path = self.imgs[index][1]
        # 读取图片，为numpy格式
        im = Image.open(img_path)
        # lbl是mat格式，mat格式的处理参考https://blog.csdn.net/qq_32425195/article/details/85202552

        # load label
        lbl = Image.open(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        lbl = torch.from_numpy(lbl)
        return self.transforms(im, lbl.unsqueeze(0))


    def __len__(self):
        '''
        返回为数据集中所有图片的个数
        :return:
        '''
        return len(self.imgs)


if __name__ == '__main__':
    pass

