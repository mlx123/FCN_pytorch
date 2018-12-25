'''调用TensorBoardX中的函数，可视化
使用方法：
在文件当前目录执行tensorboard --logdir runs

首先调用 V = Visualizer()初始化类对象，在当前目录新建了文件
然后调用V.plotxx来绘图，绘图会实时显示
绘制完所有的图像，关闭当前的对象V.close()

'''
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter


class Visualizer(object):
    def __init__(self,):  # env表示路径
        self.writer = SummaryWriter()

    def plot_many(self, tag,tensor):
        """
        用于一次绘制多张图像，每张图像需要有相同的通道数和分辨率，输入为tensor(要显示的图片数，每张图片的通道，每张图片的宽，每张图片的高)
        """
        __= vutils.make_grid(tensor)
        #with self.writer:
        self.writer.add_image(tag, __)
        self.writer.file_writer.flush()

    def plot_img(self,tag,img_tensor):
        """绘制一张图片"""
        #with self.writer:
        self.writer.add_image(tag,img_tensor)
        self.writer.file_writer.flush()
    def close(self):
        self.writer.close()

if __name__ == '__main__':
    V = Visualizer()
    _ = torch.rand(32,3,64,64)
    V.plot_many('img',_)
    V.plot_img('tag',torch.rand(3,224,224))
    V.close()