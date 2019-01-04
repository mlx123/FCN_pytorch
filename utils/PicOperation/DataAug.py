# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from SegLabelConvert import files2List
""" 
将输入图片做随机切割，即随机生成x,y坐标，然后抠出该坐标下256*256的小图，并做以下数据增强操作：

    原图和label图都需要旋转：90度，180度，270度
    原图和label图都需要做沿y轴的镜像操作
    原图做模糊操作
    原图做光照调整操作
    原图做增加噪声操作（高斯噪声，椒盐噪声）

这里我没有采用Keras自带的数据增广函数，而是自己使用opencv编写了相应的增强函数。

"""
img_w = 256
img_h = 256

image_sets = ['1.png', '2.png', '3.png', '4.png', '5.png']


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb


def creat_dataset(image_sets,picDic,labelDic,image_num=5000, mode='original'):
    """

    :param image_sets: 包含了所有image的name的的列表，用于遍历所有的image,其中pic与label的命名最好相同，方便遍历
    :param image_num: 数据增强时要生成的图片的个数
    :param mode: 除了随机裁剪，是否做数据增强
    :return:
    """
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread(os.path.join(picDic,image_sets[i])+'.BMP') # 3 channels
        label_img = cv2.imread(os.path.join(labelDic,image_sets[i])+'.png', cv2.IMREAD_GRAYSCALE)  # single channel
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)


            visualize = label_roi * 50

            cv2.imwrite(('/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/vis/r%d.png' % g_count), visualize)
            cv2.imwrite(('/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/src/r%d.BMP' % g_count), src_roi)
            cv2.imwrite(('/home/mlxuan/project/DeepLearning/data/image_Segmentation/dataAug/label/r%d.png' % g_count), label_roi)
            count += 1
            g_count += 1


if __name__ == '__main__':
    image_sets = files2List('/home/mlxuan/project/DeepLearning/data/image_Segmentation/labels')
    imageSetsBaseame = [os.path.split(image)[1].split('.')[0] for image in image_sets]#提取路径中的文件名生成新的链表
    creat_dataset(image_sets = imageSetsBaseame,
                  picDic= '/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2',
                  labelDic= '/home/mlxuan/project/DeepLearning/data/image_Segmentation/convert1',mode='augment')
