# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from collections import OrderedDict


class self_attention(nn.Module):
    def __init__(self, in_channles):
        super(self_attention, self).__init__()
        self.in_channels = in_channles

        self.f = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.softmax_ = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

        #self.init_weight(self.f)
        #self.init_weight(self.g)
        #self.init_weight(self.h)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        #assert channels == self.in_channels
        f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)

        h = self.h(x).view(batch_size, channels, -1)  # B * C * (H * W)

        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W

        return self.gamma * self_attention_map + x
    

class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        """
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('dropout1',nn.Dropout(p=0.2))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))
        self.add_module('dropout2',nn.Dropout(p=0.2))
        """
        self.add_module('conv1',nn.Conv2d(in_channels = in_channels, out_channels = growth_rate , kernel_size= 3, padding=1))
        #self.add_module('conv1',DeformConv2d(inc = in_channels, outc = growth_rate , kernel_size= 3, padding=1, stride=1, bias=None, modulation=False))
        self.add_module('dropout1',nn.Dropout(p=0.2))
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('norm1',nn.BatchNorm2d(growth_rate))
        

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))

class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


    


class Encoder(BasicModule):
    def __init__(self, growth_rate=16, block_config=([6,6,6]),
                 bn_size=64, theta=0.5, num_classes=2):
        super(Encoder, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 表示cifar-10
        if num_classes == 2:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(1, num_init_feature,
                                    kernel_size=1, stride=1,
                                    padding=0, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(1, num_init_feature,
                                    kernel_size=5, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))



        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                att_feature = num_feature//2
                self.features.add_module('attention%d' % (i + 1),
                                         self_attention(in_channles = att_feature))
                num_feature = int(num_feature * theta)

        #self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        #self.features.add_module('relu5', nn.ReLU(inplace=True))
        #self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        
        #self.features.add_module('attention', self_attention(in_channles = num_feature))
        self.attention = self_attention(in_channles = num_feature)
#         self.pool = nn.MaxPool2d(3, stride=3)
        #self.features.add_module('max_pool', nn.MaxPool2d(3, stride=3))
        self.features.add_module('dropout1',nn.Dropout(p=0.1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        features = self.features(x)
#         features = self.attention(features)
        out = features.view(features.size(0), -1)
        return out



class DCD(BasicModule):
    def __init__(self,h_features=64,input_features=64):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(6160*2,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(64,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        #out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier(BasicModule):
    def __init__(self,input_features=6160):
        super(Classifier,self).__init__()
        self.fc1 = nn.Linear(input_features,2)
        #self.fc2 = nn.Linear(32,2)

    def forward(self,input):
        #out = self.fc1(input)
        return F.softmax(self.fc1(input),dim=1)

class Classifier1(BasicModule):
    def __init__(self,input_features = 2048):
        super(Classifier1,self).__init__()
        self.dropout = nn.Dropout(p=0.5) 
        self.fc1 = nn.Linear(input_features,2)

    def forward(self,input):
        out = self.dropout(input)
        #out = self.fc1(out)
        return F.softmax(self.fc1(out),dim=1)
    
class Encoder1(BasicModule):
    def __init__(self):
        super(Encoder1,self).__init__()

        self.conv1=nn.Conv2d(1,32,3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,128,3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,256,3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5=nn.Conv2d(256,512,3)
        self.GAP= nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(32,32)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out = self.bn1(out)
        out=F.relu(self.conv2(out))
        out = self.bn2(out)      
        out=F.relu(self.conv3(out))
        out = self.bn3(out)
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv4(out))
        out = self.bn4(out)
        out=F.relu(self.conv5(out))
        out=F.max_pool2d(out,2)
        #out = self.GAP(out)
        out=out.view(out.size(0),-1)
        #print(out.shape)
        #out=self.fc1(out)

        return out






