"""
做语义分割时，有事需要使用多个数据集。不同数据集中对同一类物体的标注代码不同，需要做转换将不同的标注统一

"""
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


# this table is used to convert a lable with more tags to other label with less tags
"""
0 background -> 0 other
1 Road        ->4 Road
2 Tree        ->1 Veg(包括耕地、林地、草地)
3 Grass　　　　－>1 Veg
4 GrassAndSoil ->1 Veg
5 Soil        ->1 Veg
6 withered grass ->1 Veg
7 water        ->3 water
8 Buildings   ->2 Buildings
9 Greenhouse ->1 Veg

"""
convertTable = {0:0,1:4,2:1,3:1,4:1,5:1,6:1,7:3,8:2,9:1}
srcLabel = './DJI_0200_31.png'

def labelConvert(srcLabel,convertDict):
    """

    :param srcLabel:原标签
    :param convertDict: 用于转换ｌａｂｅｌ的字典
    :return:
    """

    s = os.path.split(srcLabel)
    fn = s[1].split('.')
    basename = fn[0]
    ext = fn[-1]

    img = Image.open(srcLabel)
    imgArr = np.asarray(img)
    print(imgArr.shape)

    #converTable[]中要根据label的通道数更改，此处时４通道的格式（ＲＧＢＡ）,其中第一通道是标注的ｔａｇ值，所以convertTable[imgArr[i][j][0]]
    imgArr2 = [[convertDict[imgArr[i][j][0]] for j in range(len(imgArr[i]))] for i in range(len(imgArr))]

    imgLable2 = Image.fromarray(np.uint8(imgArr2))
    imgLable2.save(basename+'_2.'+ext,'PNG')

    imgLableVis = Image.fromarray(np.uint8(imgArr2)*50)
    imgLableVis.save(basename+'_2Vis.'+ext,'PNG')


labelConvert(srcLabel,convertTable)