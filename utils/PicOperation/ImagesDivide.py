# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集
    将所有的srcImage和Label分别拷贝到train和label文件夹中，然后就会按照训练集、验证集、测试集的比例划分他们
    但是，当训练集和验证集数据属于不同分布时（训练集为了数据量，使用了非常大量的数据，这些数据可能与验证集、测试集的分布不完全相同），
    而验证集要最接近真实状况，里面都要是最真实的数据，即数据不饿能够随机划分，这时候需要将最接近真实的那部分数据随机抽出一部分给train,
    剩下的那部分作为val和test
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
            imgs_list= []
            imgs_list = glob.glob(os.path.join(root, sDir) + '/*.png')
            imgs_list = imgs_list if imgs_list else glob.glob(os.path.join(root, sDir) + '/*.BMP')
            imgs_list = sorted(imgs_list,key=str.lower)
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


def gen_txt(txtPath, imgDir):
    """

    :param 给定要产生的txt文件的路径
    :param 给定要遍历的图片的路径
    :结果: 将要遍历的图片路径写入给定的txt文档中
    """
    f = open(txtPath, 'w')

    iDir = os.path.join(imgDir)  # 获取各类的文件夹 绝对路径
    img_list = os.listdir(iDir)  # 获取类别文件夹下所有png图片的路径
    for i in range(len(img_list)):
        if not (img_list[i].endswith('png') or img_list[i].endswith('BMP')):  # 若不是png文件，跳过
            continue
        # label = img_list[i].split('_')[0]
        # img_path = os.path.join(i_dir, img_list[i])
        line = img_list[i] + '\n'
        f.write(line)

    f.close()


def picFulPath(txtPath, rootImg, rootLbl,
               destPath,
               ImgFix = '',
               lblFix = ''
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
        line = rootImg + line + ' ' + rootLbl + line.split('.')[0]+'.png'  + '\n'  # 此处最好改为文件名的相加，而不是单纯的字符串相加
        f.write(line)
    f.close()
    fh.close()





if __name__ == '__main__':

    """
    #用法
    指定各个文件夹：
    数据所在文件夹dataset_dir，该文件夹下同时包含了几个文件夹 分别存放原图img和标准图label和可视图vis
    要生成的训练集 验证集 测试集的文件夹 train_dir valid_dir  test_dir
    dataset_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/'
    train_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/train/'
    valid_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/valid/'
    test_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/test/'
    
    #制定训练集 验证集 测试集 的划分比例
    train_per = 0.8
    valid_per = 0.2
    
    #图片划分
    splitPicture(dataset_dir, train_dir, valid_dir, test_dir, 0.8, 0.2)
    
    
    train_txt_path = '../Data/train.txt'
    valid_txt_path = '../Data/valid.txt'

    #这些代码只用执行一次，用来生成img lbl的txt文档
    
    """
    """
    dataset_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/'
    train_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/'
    valid_dir = '/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/'
    test_dir =  '/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/test/'
    train_per = 0.9
    valid_per = 0.1
    splitPicture(dataset_dir, train_dir, valid_dir, test_dir, 0.9, 0.1)
    """
    #遍历给定目录下的所有的png文件和BMP文件，生成对应的txt
    gen_txt(txtPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/train.txt', imgDir='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/src')
    gen_txt(txtPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/valid.txt', imgDir='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/src')

    #根据目录名 生成img+label的dataAug/完全路径
    picFulPath(txtPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/train.txt',
               rootImg ='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/src/',
               rootLbl='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/label/',
               destPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/train/trainFull.txt')

    picFulPath(txtPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/valid.txt',
               rootImg='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/src/',
               rootLbl='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/label/',
               destPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/valid/validFull.txt')
