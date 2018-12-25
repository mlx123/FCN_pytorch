import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):

    pretrained_model = \
        osp.expanduser('/home/mlxuan/project/DeepLearning/pretrained/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vM2oya3k0Zlgtekk',
            path=cls.pretrained_model,
            md5='8acf386d722dc3484625964cbe2aba49',
        )



    def __init__(self, n_class=21):
        super(FCN32s, self).__init__()
        #将每个层都定义为FCN32s类的参数
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()


        #用于计算每个类的分值
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        #ConvTranspose2d是Conv2d的逆操作，此处输入通道数和输出通道数都是n_class，上采样卷积核为64，步进为32.上采样反卷积的操作参考github
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            #一般是采用预训练的Vgg16的参数初始化卷积层和全连接层，所以此处的初始化没有用
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            #上采样层一般是一种操作，应该没有操作，如双线性差值，最近邻差值等
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x):

        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)

        #只要最后的的x.size()的行列，舍去了最前面的19行，19列
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h


    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())


if __name__ == '__main__':

    """import torchvision
    def VGG16(pretrained=False):
        model = torchvision.models.vgg16(pretrained=False)
        if not pretrained:
            return model
        model_file = _get_vgg16_pretrained_model()
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
        return model


    def _get_vgg16_pretrained_model():
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
            path=osp.expanduser('/home/mlxuan/project/DeepLearning/pretrained/vgg16_from_caffe.pth'),
            md5='aa75b158f4181e7f6230029eb96c1b13',
        )



    # 可视化权重和网络图,
    #卷积层可视化基本上看不出什么来，先不深究
    net = FCN32s()
    # vgg16 = VGG16(pretrained=True)
    # net.copy_params_from_vgg16(vgg16)
    from tensorboardX import SummaryWriter
    import torchvision.utils as vutils
    writer = SummaryWriter()
    params  = net.state_dict()
    
    for k,v in params.items():
        if 'conv' in k and 'weight' in k:
            c_int = v.size()[1]
            c_out = v.size()[0]
            for j in range(c_out):
                print(k, v.size(), j)
                kernel_j = v[j, :, :, :].unsqueeze(1)  # 压缩维度，为make_grid制作输入
                kernel_grid = vutils.make_grid(kernel_j, normalize=True, scale_each=True, nrow=8)  # 1*输入通道数, w, h
                writer.add_image(k + '_split_in_channel', kernel_grid, global_step=j)
            # k_w,k_h = v.size()[-1],v.size()[-2]
            # kernel_all = v.view(-1,1,k_w,k_h)
            # kernel_grid = vutils.make_grid(kernel_all,normalize=True,scale_each=True,nrow=c_int)
            # writer.add_image(k+'all',kernel_grid,global_step=666)
    writer.close()
    
    resnet18 = torchvision.models.resnet18(False)
    writer.add_graph(resnet18,torch.rand(6,3,224,224))
    writer.add_graph(net,torch.rand(6,3,224,224))
    writer.close()
    #使用部分图片测试，并且可视化feature map
    net = FCN32s()
    from tensorboardX import SummaryWriter
    import torchvision.utils as vutils
    writer = SummaryWriter()
    writer.add_graph(net, torch.rand(6, 3, 224, 224))
    writer.close()
    """
    net = FCN32s()
    A = net.modules()


    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            print(m.bias)
            print(m.weight)
    '''
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)'''