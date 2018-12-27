# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集
"""

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

dataset_dir = '/home/mlxuan/project/DeepLearning/data/remoteSensing/Segmentation/1/train'
train_dir = '../Data/train/'
valid_dir = '../Data/valid/'
test_dir = '../Data/test/'

train_per = 0.8
valid_per = 0.2

train_txt_path = '../Data/train.txt'
valid_txt_path = '../Data/valid.txt'







def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def splitPicture(datasetDir,trainDir,valDir,testDir,trainPer=0.8,valPer=0.2):
    """
    将rootPic路径下的pic和label图片按训练集、验证机和测试集分类，一般语义分割时，rootPic路径下包含label，pic子目录（分别存储语义分割后的标签图片和原图）
        要点：
        1.os.walk()的作用:os.walk后面跟目录的路径(记为主目录)，生成一个生成器
        每次返回是一个tuple,(这次遍历的目录路径是什么(字符串)，这次遍历的目录下的子目录（列表形式），这次遍历的目录下的文件)
        依次遍历主目录下的每个子目录，直到遍历完所有的子目录为止
        对于目录的操作基本都是用os包，如os.path.join进行目录的连接
        2.glob.glob()  后跟字符串，代表想要匹配的字符串形式
        返回一个列表，列表包含了所有和参数匹配的文件的路径

        3.random
        random根据种子的值进行随机，当种子相同时，random的随机值也相同
        """

    for root, dirs, files in os.walk(datasetDir):
        for sDir in dirs:
            imgs_list = glob.glob(os.path.join(root, sDir) + '/*.png')
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            train_point = int(imgs_num * trainPer)
            valid_point = int(imgs_num * (trainPer + valPer))

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = trainDir + sDir + '/'
                elif i < valid_point:
                    out_dir = valDir + sDir + '/'
                else:
                    out_dir = testDir + sDir + '/'

                makedir(out_dir)
                out_path = out_dir + os.path.split(imgs_list[i])[-1]
                shutil.copy(imgs_list[i], out_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sDir, train_point, valid_point - train_point,
                                                                 imgs_num - valid_point))


def gen_txt(txt_path, img_dir):
    """

    :param 给定要产生的txt文件的路径
    :param 给定要遍历的图片的路径
    :return: 将要遍历的图片路径写入给定的txt文档中
    """
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取给定文件下（一般给定文件夹为train valid test,表示不同作用的数据集）各子文件夹名称，一般需要包含2个子文件夹label,src，表示标签数据和图片源数据
        for sub_dir in s_dirs:

            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                # label = img_list[i].split('_')[0]
                # img_path = os.path.join(i_dir, img_list[i])
                line = img_list[i] + '\n'
                f.write(line)
            break
    f.close()


def picFulPath(txtPath, rootImg, rootLbl,
               destPath,
               ImgFix,
               lblFix
               ):
    '''
    给定只包含图片文件名的txt和图片所在路径
    参数：txtPath  包含图片文件名的txt的路径
         rootImg:图片所在的根文件夹
         rootLbl:标记文件所在的根目录
         destPath:要生成的包含img和lbl完全路径的txt文档
    :return: 包含图片绝对路径和图片文件名、包含标签绝对和标签和的txt文档

    '''
    # 读取txtpath中的每一行，然后加上rootImag和rootLbl后写入新的txt文档
    f = open(destPath, 'w')
    fh = open(txtPath, 'r')
    # 读取txtPath中的每一行，每一行都是图片文件的文件名，在每一行中加上绝对路径
    for line in fh:
        line = line.rstrip()  # 去掉改行最后的回车符号
        line = rootImg + line + ' ' + rootLbl + line  + '\n'  # 此处最好改为文件名的相加，而不是单纯的字符串相加
        f.write(line)
    f.close()
    fh.close()




class RSDataClassSeg(data.Dataset):
    class_names = np.array([
        'background',
        'vegetation',
        'building',
        'water',
        'road' ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])


    def __init__(self, txt_path, transforms=None, train=True):
        '''
         参数：txt_path包含了每张图片的路径和图片的标签的路径（语义分割时就是标记好的图片的路径）
             transforms表示该读取图片后，图片要进行的预处理，如果为None,则使用默认的预处理
         '''

        # 1.指定数据集，常用的方法是给定图片路径的txt文档,将他们读入一个列表
        imgs = []
        self.train = train  # 表示此次时训练集还是数据集，可能需要用来做不同的数据预处理
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
                # T.Resize T.CenterCrop将图片保持纵横比缩放裁剪为同一大小，此处图片已经预先裁剪为指定的大小，所以不需要裁剪
                # T.Tensor,T.Normalize 将图片转为[0,1]的Tensor,并归一化
                if train == True:
                    transImg = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([T.ToPILImage(),
                                          T.ToTensor()])
                else:
                    transImg = T.Compose([T.ToTensor(),
                                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([T.ToPILImage(), T.ToTensor()])

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

                return ((255 * imgUntrans).type(torch.ByteTensor).transpose(0, 1).transpose(1, 2)).numpy(), (
                    lbl.squeeze(0)).numpy()

            # 常规的数据操作：（裁剪为统一大小,可选T.scale），（数据增强，如随机裁剪等 语义分割时一般不做这个），ToTensor()后+T.Normalize
            # if self.train:  # 如果此次是训练集（训练集和验证集可能读取数据方法一样，但是预处理的过程不一样）
        self.transforms = transforms
        self.untransform = untransforms
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

    #这些代码只用执行一次，用来生成img lbl的txt文档
    splitPicture(dataset_dir, train_dir, valid_dir, test_dir, 0.8, 0.2)
    gen_txt(train_txt_path, train_dir)
    gen_txt(valid_txt_path, valid_dir)

    picFulPath(train_txt_path,'/home/mlxuan/project/DeepLearning/FCN/pytorch-fcn-wh2/Data/train/src/','/home/mlxuan/project/DeepLearning/FCN/pytorch-fcn-wh2/Data/train/label/','../Data/trainFullPath.txt','','')
    picFulPath(valid_txt_path,'/home/mlxuan/project/DeepLearning/FCN/pytorch-fcn-wh2/Data/valid/src/','/home/mlxuan/project/DeepLearning/FCN/pytorch-fcn-wh2/Data/valid/label/','../Data/validFullPath.txt','','')
    

