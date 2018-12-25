'''定义自己的数据集'''
# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import scipy.io
import  torch
import utils


def picFulPath(txtPath,rootImg,rootLbl,
               destPath='/home/mlxuan/project/DeepLearning/FCN/fcn_mlx/data/ImagAndLal.txt',
               ImgFix = '.jpg',
               lblFix = '.mat'
               ):
    '''
    给定只包含图片文件名的txt和图片所在路径
    参数：txtPath  包含图片文件名的txt的路径  
         rootImg:图片所在的根文件夹
         rootLbl:标记文件所在的根目录
    :return: 包含图片绝对路径和图片文件名、包含标签绝对和标签和的txt文档
    
    '''
    #读取txtpath中的每一行，然后加上rootImag和rootLbl后写入新的txt文档
    f = open(destPath,'w')
    fh = open(txtPath, 'r')
    #读取txtPath中的每一行，每一行都是图片文件的文件名，在每一行中加上绝对路径
    for line in fh:
        line = line.rstrip()  # 去掉改行最后的回车符号
        line = rootImg+line+ImgFix+' '+rootLbl+line+lblFix+'\n'#此处最好改为文件名的相加，而不是单纯的字符串相加
        f.write(line)
    f.close()
    fh.close()


class SBDClassSeg(data.Dataset):

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

    def __init__(self, txt_path, transforms=None,train=True):#此处train val test（指定此次的数据时用来做什么的，不同用处的数据集图片预处理方法不同，可以传入不同的txt_path即可）的指定应该用union

        '''
        dataset是为了dataloader而定义，dataloader是为了按照batch遍历数据集，所以dataset要制定数据集、获得数据集元素的方法，数据集长度
        dataset生成self.imgs,这是一个list,包含了数据集的所有图片组成的list
        参数：txt_path包含了每张图片的路径和图片的标签（语义分割时就是标记好的图片的路径）
            transforms表示该读取图片后，图片要进行的预处理，如果为None,则使用默认的预处理
        '''

        #1.指定数据集，常用的方法是给定图片路径的txt文档,将他们读入一个列表
        imgs=[]
        self.train=train
        fh = open(txt_path, 'r')
             #遍历整个txt文档，每张图片的路径作为list的一个元素
        for line in fh:
            line = line.rstrip()#去掉改行最后的回车符号
            words = line.split()
            imgs.append((words[0], words[1]))#words[0]是原图片的路径，words[1]是标记好的图片的标签
        self.imgs = imgs
        # 2.指定读入图片后的预处理
        if transforms is None:

            def transforms(img,lbl):
                if train == True:
                    #T.Resize T.CenterCrop将图片保持纵横比缩放裁剪为同一大小，
                    #T.Tensor,T.Normalize 将图片转为[0,1]的Tensor,并归一化
                    transImg = T.Compose([T.Resize(224),T.CenterCrop(224),
                                   T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    #lbl时mat格式，处理参考https://blog.csdn.net/qq_32425195/article/details/85202552
                    #T.ToPILImage()将tensor转为PILImage，然后做Resize和CenterCrop处理
                    transLbl = T.Compose([T.ToPILImage(),T.Resize(224),T.CenterCrop(224),
                                   T.ToTensor()])
                else:
                    transImg = T.Compose([T.ToTensor(),
                                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    transLbl = T.Compose([T.ToPILImage(), T.ToTensor()])

                img = transImg(img)
                lbl = transLbl(lbl)
                return img,lbl.long()

            def untransforms(img,lbl):
                #该trans是对已经归一化的tensor处理，得到归一化之前的Tensor
                trans = T.Compose(
                    [T.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])])
                imgUntrans = trans(img)
                # for i in range(img.shape[0]):
                #     imgUntrans[i] = trans(img[i])

                return ((255 * imgUntrans).type(torch.ByteTensor).transpose(0, 1).transpose(1, 2)).numpy(), (
                    lbl.squeeze(0)).numpy()

            # 常规的数据操作：（裁剪为统一大小,可选T.scale），（数据增强，如随机裁剪等 语义分割时一般不做这个），ToTensor()后+T.Normalize
            if self.train:#如果此次是训练集（训练集和验证集可能读取数据方法一样，但是预处理的过程不一样）
                self.transforms = transforms
                self.untransforms = untransforms
        fh.close()

    def __getitem__(self, index):#该方法是给定索引或键值，返回对应的值，常用在enumerate遍历数据集时
        '''返回一张图片的验证集和测试集的数据'''
        #从list中获取图片的路径
        img_path = self.imgs[index][0]
        lbl_path = self.imgs[index][1]
        #读取图片，为numpy格式
        im = Image.open(img_path)
          #lbl是mat格式，mat格式的处理参考https://blog.csdn.net/qq_32425195/article/details/85202552

        mat = scipy.io.loadmat(lbl_path)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)

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
    picFulPath('/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/train.txt','/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/img/','/home/mlxuan/project/DeepLearning/data/benchmark/benchmark_RELEASE/dataset/cls/')
    train_dataset = SBDClassSeg('./ImagAndLal.txt')
    from torch.utils.data import DataLoader
    trainloader = DataLoader(train_dataset,batch_size=2,shuffle=False,drop_last=True)
    V = utils.Visualizer()
    #按batch_size遍历trainloader
    for i,(data,label) in enumerate(trainloader):# 如果batch_size不是1，那么每次输入的batch_size个Tensor就需要有相同的size，否则就不能遍历。还可以用for data,label in trainloader:遍历
        data,label = trainloader.dataset.untransforms(data,label)

        V.plot_many('imgs'+str(i),data)
        V.plot_img('imgs' + str(i), label)

    V.close()

