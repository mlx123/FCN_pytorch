import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from SegLabelConvert import files2List

#将给定的图片按照给定的行列数切割  保存时将原始数据保存为bmp格式，可以防止数据保存时失真。如果保存为ｊｐｇ格式，则数据有压缩丢失等
def SplitImage(src, rownum, colnum, dstpath,dstTxtPath,Prefix = ''):
    img = Image.open(src)
    imgArray = np.asarray(img)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')

        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]

        num = 0
        rowheight = h // rownum
        colwidth = w // colnum

        f = open(dstTxtPath,'a')
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                ext = ext if ext !='JPG' else 'BMP'
                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                f.write('\"'+Prefix+basename + '_' + str(num) + '.' + ext+'\",'+'\n')
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)
        f.close()
    else:
        print('不合法的行列切割参数！')



def Split2Images(src,label,rownum,colnum,dstImagepath,dstLabelPath,dstImageTxtPath,dstLabelImagePath,dstImagePrefix='',dstLabelPrefix=''):
    img = Image.open(src)
    wSrc, hSrc = img.size
    img = Image.open(label)
    wLabel, hLabel = img.size

    if(wSrc == wLabel and hSrc ==hLabel):
        SplitImage(src,rownum,colnum,dstpath=dstImagepath,dstTxtPath=dstImageTxtPath,Prefix=dstImagePrefix)
        SplitImage(label,rownum,colnum,dstpath= dstLabelPath,dstTxtPath=dstLabelImagePath,Prefix=dstImagePrefix)
    else:
        print('Image and label are not match ')


#图像拼接
def SpliceImages(imagePathSets,rows,cols,dstPath):
    """

    :param imageSets: 要拼接的图片的路径组成的列表
    :param rows: 一列的图片数目
    :param cols: 一行的图片数目
    :param dstPath: 拼接后的图片所在的路径
    :return:
    """
    assert len(imagePathSets)==rows*cols,'len(imageSets) must equal to rows*cols'

    imageSets = list(map(Image.open,imagePathSets))#函数式变成，map(f,列表)将f作用于列表的每一个元素后生成新的列表

    target = Image.new('RGBA', (4000,3000))#此处代码需要改进 改进到能够自动计算拼接后图片的大小 而不是人工计算后赋值
    for i in range(rows):
        for j in range(cols):
            target.paste(imageSets[i*rows+j],(j*400,i*300,j*400+400,i*300+300))

    target.save('./spliceDJI0287.png')




if __name__ == '__main__':
    imageSets = []
    with open('/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split/0287.txt','r') as fh:
        for line in fh:
            line = line.rstrip()
            imageSets.append(line)

    SpliceImages(imageSets,10,10,'')

    """SplitImage(src='/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/DJI_0214.JPG',
               rownum = 2, colnum = 2,
               dstpath ='/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2000*1500/',
               dstTxtPath='/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2000*1500/1.txt')

  
Split2Images('/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/DJI_0200.JPG',
         '/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/annotations/DJI_0200.png',
         10,10,
         '/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2',
         '/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2',
         '/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2/image.txt',
         '/home/mlxuan/project/DeepLearning/data/image_Segmentation/js-segment-annotator-master/data/images/Split2/label.txt',
         'data/images/Split/',
         'data/images/Split/'
         )

"""

"""src = input('请输入图片文件路径：')
if os.path.isfile(src):
dstpath = input('请输入图片输出目录（不输入路径则表示使用源图片所在目录）：')
if (dstpath == '') or os.path.exists(dstpath):
    row = int(input('请输入切割行数：'))
    col = int(input('请输入切割列数：'))
    if row > 0 and col > 0:
        SplitImage(src, row, col, dstpath)
    else:
        print('无效的行列切割参数！')
else:
    print('图片输出目录 %s 不存在！' % dstpath)
else:
print('图片文件 %s 不存在！' % src)
"""
