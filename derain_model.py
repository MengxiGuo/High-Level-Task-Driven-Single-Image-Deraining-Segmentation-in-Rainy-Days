from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import h5py
from PIL import Image
import functools
import os
import time
import math
import random
import hashlib
from spectral import SpectralNorm

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, outputplanes, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, outputplanes // 2, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, outputplanes // 2, 3, padding=dilations[1], dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, outputplanes // 2, 3, padding=dilations[2], dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, outputplanes // 2, 3, padding=dilations[3], dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outputplanes // 2, 1, stride=1, bias=False),
                                             BatchNorm(outputplanes//2),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d((outputplanes * 5) // 2, outputplanes, 1, bias=False)
        self.bn1 = BatchNorm(outputplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):

        x1 = self.aspp1(x)

        x2 = self.aspp2(x)

        x3 = self.aspp3(x)

        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        output = self.dropout(x)
        # print(x.size())
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP_Musk(nn.Module):
    def __init__(self, inplanes, outputplanes, output_stride, BatchNorm):
        super(ASPP_Musk, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, outputplanes // 2, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, outputplanes // 2, 3, padding=dilations[1], dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, outputplanes // 2, 3, padding=dilations[2], dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, outputplanes // 2, 3, padding=dilations[3], dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outputplanes // 2, 1, stride=1, bias=False),
                                             BatchNorm(outputplanes//2),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d((outputplanes * 5) // 2, outputplanes, 1, bias=False)
        self.bn1 = BatchNorm(outputplanes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):

        x1 = self.aspp1(x)

        x2 = self.aspp2(x)

        x3 = self.aspp3(x)

        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        output = self.dropout(x)
        # print(x.size())
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(inplanes, outputplanes, output_stride, BatchNorm):
    return ASPP(inplanes, outputplanes, output_stride, BatchNorm)


def build_aspp_musk(inplanes, outputplanes, output_stride, BatchNorm):
    return ASPP_Musk(inplanes, outputplanes, output_stride, BatchNorm)

class LaPulas_Fliter(nn.Module):
    def __init__(self):
        super(LaPulas_Fliter, self).__init__()
        kernel = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=1, padding=1)
        x = self.relu(x)
        return x


class conv_LSTM(nn.Module):
    def __init__(self):
        super(conv_LSTM, self).__init__()
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, h, c):
        x = torch.cat((x, h), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvBNReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBNReLU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = SpectralNorm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNReLU_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvBNReLU_1, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = SpectralNorm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResBlock, self).__init__()
        self.conv0 = ConvBNReLU(in_dim, in_dim)
        self.conv1 = ConvBNReLU(in_dim, in_dim)

    def forward(self, x):
        z = self.conv0(x)
        z = self.conv1(z)
        output = F.relu(z + x)
        return output


class RepatResBlock(nn.Module):
    def __init__(self, in_dim):
        super(RepatResBlock, self).__init__()
        self.resblock = ResBlock(in_dim)

    def forward(self, x):
        for i in range(5):
            x = self.resblock(x)
        return x


class PreNet_LSTM(nn.Module):
    def __init__(self, iteration):
        super(PreNet_LSTM, self).__init__()
        self.iteration = iteration
        self.conv_in = nn.Sequential(nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU())  # torch.cat(xt0,xt1,xb)
        self.LSTM = conv_LSTM()
        self.res_block1 = ResBlock(32)
        self.res_block2 = ResBlock(32)
        self.res_block3 = ResBlock(32)
        self.res_block4 = ResBlock(32)
        self.res_block5 = ResBlock(32)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        input = x
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        x_list = []
        for itera in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv_in(x)
            h, c = self.LSTM(x, h, c)
            x = h
            x = self.res_block1(x)
            x = self.res_block2(x)
            x = self.res_block3(x)
            x = self.res_block4(x)
            x = self.res_block5(x)
            x = self.conv_out(x)
            x_list.append(x)
        return x_list


'''
class PreNet_r(nn.Module):
    def __init__(self, layer_number):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(6, 32, kernel_size=1,stride=1)#torch.cat(xt0,xt1,xb)
        self.LSTM = conv_LSTM()
        self.medial_layer = []
        for i in range(layer_number-2):
            self.medial_layer.append(RepatResBlock(32).cuda())
        self.conv_out = nn.Conv2d(32, 3, kernel_size=1, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv_in(x)
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        for i in range(len(self.medial_layer)):
            h, c = self.LSTM(x,h,c)
            x = h
            x = self.medial_layer[i](x)
        x = self.conv_out(x)
        return x
'''


class Adversary(nn.Module):
    def __init__(self, in_channels):  # message_length is default in lua file,the value 30
        super(Adversary, self).__init__()
        self.in_channels = in_channels
        # if grayscale,input dimension batchsize*1*H*W;else,batchsize*3*H*W	note from lua file
        self.conv1 = ConvBNReLU_1(in_channels, 64)
        self.conv2 = ConvBNReLU_1(64, 64)
        self.conv3 = ConvBNReLU_1(64, 64)
        self.average_pooling = nn.AdaptiveAvgPool2d(
            1)  # nn.SpatialAdaptiveAveragePooling(1, 1)) in lua file; 2D or 3D? batchsize*L*1*1
        self.linear = SpectralNorm(nn.Linear(64, 1))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = self.sigmoid(x)
        return x


#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class downsample_unit(nn.Module):
    def __init__(self, indim, outdim):
        super(downsample_unit, self).__init__()
        downsample_list = [nn.Conv2d(indim, outdim, kernel_size=3, stride=2, padding=1),
                           nn.BatchNorm2d(outdim), nn.LeakyReLU(0.2)]
        self.model = nn.Sequential(*downsample_list)

    def forward(self, x):
        return self.model(x)


class upsample_unit(nn.Module):
    def __init__(self, indim, outdim):
        super(upsample_unit, self).__init__()
        upsample_list = [nn.ConvTranspose2d(indim, outdim, kernel_size=3, stride=2, padding=1, output_padding=1),
                         nn.BatchNorm2d(outdim), nn.ReLU(True)]
        self.model = nn.Sequential(*upsample_list)

    #     def forward(self, x, skip_input):
    #         x = self.model(x)
    #         if skip_input==None:
    #             return x
    #         else:
    #             x = torch.cat((x, skip_input), 1)
    #             return x

    def forward(self, x):
        x = self.model(x)
        return x


class Derain_GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', requires_grad= True):
        assert (n_blocks >= 0)
        super(Derain_GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        init_unit = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                     activation]
        self.init_conv_unit = nn.Sequential(*init_unit)

        ### downsample
        # downsample_list = []
        # for i in range(n_downsampling):
        #     mult = 2**i
        #     downsample_list += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #               norm_layer(ngf * mult * 2), activation]
        # self.downsample_seq =  nn.Sequential(*downsample_list)
        # setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))

        for i in range(n_downsampling):
            mult = 2 ** i
            setattr(self, 'down' + str(i), downsample_unit(ngf * mult, ngf * mult * 2))  # in  out

        ### resnet blocks
        mult = 2 ** n_downsampling
        resblock_list = []
        for i in range(n_blocks):
            resblock_list += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblock_seq = nn.Sequential(*resblock_list)

        ### upsample
        # upsample_list=[]
        # for i in range(n_downsampling):
        #     mult = 2**(n_downsampling - i)
        #     upsample_list += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
        #                norm_layer(int(ngf * mult / 2)), activation]
        # self.upsample_seq = nn.Sequential(*upsample_list)
        #
        # for i in range(n_downsampling):
        #     mult = 2**(n_downsampling - i)
        #     setattr(self, 'up'+str(i), upsample_unit(ngf * mult,  int(ngf * mult / 2)) )

        mult = 2 ** (n_downsampling - 0)
        setattr(self, 'up' + str(0), upsample_unit(ngf * mult, int(ngf * mult / 2)))
        mult = 2 ** (n_downsampling - 1)
        setattr(self, 'up' + str(1), upsample_unit(ngf * mult * 2, int(ngf * mult / 2)))
        mult = 2 ** (n_downsampling - 2)
        setattr(self, 'up' + str(2), upsample_unit(ngf * mult * 2, int(ngf * mult / 2)))
        mult = 2 ** (n_downsampling - 3)
        setattr(self, 'up' + str(3), upsample_unit(ngf * mult, int(ngf * mult / 2)))

        output_list = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.out_unit = nn.Sequential(*output_list)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        x_init = self.init_conv_unit(input)
        down_unit0 = getattr(self, 'down0');
        d0 = down_unit0(x_init)
        down_unit1 = getattr(self, 'down1');
        d1 = down_unit1(d0)
        down_unit2 = getattr(self, 'down2');
        d2 = down_unit2(d1)
        down_unit3 = getattr(self, 'down3');
        d3 = down_unit3(d2)

        res = self.resblock_seq(d3)
        # print('resnet size:', res.shape, '\n')  # if input is 256*256 , then resnet output is 16*16

        up_uint0 = getattr(self, 'up0');
        up0 = up_uint0(res)
        up0 = torch.cat((up0, d2), 1)
        up_uint1 = getattr(self, 'up1');
        up1 = up_uint1(up0);

        up1 = torch.cat((up1, d1), 1)
        up_uint2 = getattr(self, 'up2');
        up2 = up_uint2(up1);

        up_uint3 = getattr(self, 'up3');
        up3 = up_uint3(up2)

        out = self.out_unit(up3)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


####create generator
def create_gen_nets():
    generator = Derain_GlobalGenerator(input_nc=4, output_nc=3, ngf=16, n_downsampling=4, n_blocks=9,
                                       norm_layer=nn.BatchNorm2d,
                                       padding_type='reflect')

    if torch.cuda.is_available():
        generator = generator.cuda()

    generator.apply(weights_init_normal)
    # print_network(generator)

    return generator


######################################################################
######################################################################
######################################################################
######################################################################
class Discriminator_n_layers(nn.Module):
    def __init__(self, args):
        super(Discriminator_n_layers, self).__init__()

        n_layers = args.n_D_layers
        in_channels = args.out_channels

        def discriminator_block(in_filters, out_filters, k=4, s=2, p=1, norm=True, sigmoid=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=k, stride=s, padding=p)]
            if norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sigmoid:
                layers.append(nn.Sigmoid())
                print('use sigmoid')
            return layers

        sequence = [*discriminator_block(in_channels * 2, 64, norm=False)]  # (1,64,128,128)

        assert n_layers <= 5

        if (n_layers == 1):
            'when n_layers==1, the patch_size is (16x16)'
            out_filters = 64 * 2 ** (n_layers - 1)

        elif (1 < n_layers & n_layers <= 4):
            '''
            when n_layers==2, the patch_size is (34x34)
            when n_layers==3, the patch_size is (70x70), this is the size used in the paper
            when n_layers==4, the patch_size is (142x142)
            '''
            for k in range(1, n_layers):  # k=1,2,3
                sequence += [*discriminator_block(2 ** (5 + k), 2 ** (6 + k))]
            out_filters = 64 * 2 ** (n_layers - 1)

        elif (n_layers == 5):
            '''
            when n_layers==5, the patch_size is (286x286), lis larger than the img_size(256),
            so this is the whole img condition
            '''
            for k in range(1, 4):  # k=1,2,3
                sequence += [*discriminator_block(2 ** (5 + k), 2 ** (6 + k))]
                # k=4
            sequence += [*discriminator_block(2 ** 9, 2 ** 9)]  #
            out_filters = 2 ** 9

        num_of_filter = min(2 * out_filters, 2 ** 9)

        sequence += [*discriminator_block(out_filters, num_of_filter, k=4, s=1, p=1)]
        sequence += [*discriminator_block(num_of_filter, 1, k=4, s=1, p=1, norm=False, sigmoid=True)]

        self.model = nn.Sequential(*sequence)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        # print("self.model(img_input):  ",self.model(img_input).size())
        return self.model(img_input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, inputA):
        # input = torch.cat((inputA, inputB), 1)
        input = inputA
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 3):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


## for disc loss
class Disc_MultiS_Scale_Loss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(Disc_MultiS_Scale_Loss, self).__init__()
        self.Tensor = tensor
        self.loss = nn.L1Loss().cuda()

    def __call__(self, disc_inter_feat_list, realB_MultiScale_list):
        indx = 0
        n_disc_layer = 5  # cmt
        for i in range(1, len(disc_inter_feat_list) - 2):
            #             print('len feat list:', len(disc_inter_feat_list))
            #             print('gt:real_B: ', realB_MultiScale_list[i].size() )
            #             print('inter_feat{}:'.format(i), disc_inter_feat_list[i].size() )
            weight = 2 ** (5 - indx)
            weight = 1 / weight
            # print('{} weight: '.format(i), weight)
            # print(realB_MultiScale_list[i].shape, disc_inter_feat_list[i].shape )
            if i == 1:
                loss = weight * self.loss(realB_MultiScale_list[i], disc_inter_feat_list[i])
            else:
                loss = loss + weight * self.loss(realB_MultiScale_list[i], disc_inter_feat_list[i])
            indx += 1
            # print(loss)
        return loss


def create_disc_nets():
    # discriminator = Discriminator_n_layers(args)
    discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=5,
                                        norm_layer=nn.BatchNorm2d,
                                        use_sigmoid=True, getIntermFeat=True)

    if torch.cuda.is_available():
        discriminator = discriminator.cuda()

    discriminator.apply(weights_init_normal)
    # print_network(discriminator)

    return discriminator


###############################################################

class rain_drop_musk_net(nn.Module):
    def __init__(self, n_blocks, requires_grad=True):
        super(rain_drop_musk_net, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, 1, 1)
        resblock_list = []
        for i in range(n_blocks):
            resblock_list += [
                ResnetBlock(32, padding_type='reflect', activation=nn.ReLU(True), norm_layer=nn.BatchNorm2d)]
        self.resblock_seq = nn.Sequential(*resblock_list)
        self.conv_musk = nn.Conv2d(32, 1, 3, 1, 1)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv0(x)
        x = self.resblock_seq(x)
        out = self.conv_musk(x)
        return out


####################################################################
####################################################################
####################################################################
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Partialconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv_0 = nn.Conv2d(1, 1, kernel_size, stride, padding, dilation, groups, False)
        self.mask_conv_1 = nn.MaxPool2d(kernel_size, stride, padding, dilation, False, False)
        self.weight = self.input_conv.weight
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv_0.weight, 1.0)
        for param in self.mask_conv_0.parameters():
            param.requires_grad = False
        self.kernel_size = kernel_size

    def forward(self, input, neg_mask):
        output = self.input_conv(input * neg_mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv_0(neg_mask)

        output_max_mask = self.mask_conv_1(neg_mask)
        no_update_rainy = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_rainy, 1.0)
        output_pre = (
                             output - output_bias) * output_max_mask * self.kernel_size * self.kernel_size / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_rainy, 0.0)

        new_mask = output_max_mask

        return output, new_mask


class Partial_downsample_unit(nn.Module):
    def __init__(self, indim, outdim):
        super(Partial_downsample_unit, self).__init__()
        self.P_conv = Partialconv(indim, outdim, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(True)

    def forward(self, input, neg_mask):
        h, h_mask = self.P_conv(input, neg_mask)
        h = self.bn(h)
        h = self.relu(h)
        return h, h_mask


class PCNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(PCNet, self).__init__()
        activation = nn.ReLU(True)

        self.P_init_ReflectionPad = nn.ReflectionPad2d(3)
        self.Partail_init_conv = Partialconv(input_nc, ngf, kernel_size=7, padding=0)
        self.P_init_norm = norm_layer(ngf)
        self.P_init_activarion = activation

        init_unit = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc + 1, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                     activation]
        self.init_conv_unit = nn.Sequential(*init_unit)

        ### downsample
        # downsample_list = []
        # for i in range(n_downsampling):
        #     mult = 2**i
        #     downsample_list += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #               norm_layer(ngf * mult * 2), activation]
        # self.downsample_seq =  nn.Sequential(*downsample_list)
        # setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))

        for i in range(n_downsampling):
            mult = 2 ** i
            setattr(self, 'P_down' + str(i), Partial_downsample_unit(ngf * mult, ngf * mult * 2))  # in  out

        for i in range(n_downsampling):
            mult = 2 ** i
            setattr(self, 'down' + str(i), downsample_unit(ngf * mult, ngf * mult * 2))  # in  out

        ### resnet blocks
        mult = 2 ** (n_downsampling + 1)
        resblock_list = []
        for i in range(n_blocks):
            resblock_list += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblock_seq = nn.Sequential(*resblock_list)

        ### upsample

        mult = 2 ** (n_downsampling + 1)
        setattr(self, 'up' + str(0), upsample_unit(ngf * mult, int(ngf * mult / 2)))
        mult = 2 ** (n_downsampling + 0)
        setattr(self, 'up' + str(1), upsample_unit(ngf * mult * 2, int(ngf * mult / 2)))
        mult = 2 ** (n_downsampling - 1)
        setattr(self, 'up' + str(2), upsample_unit(ngf * mult * 2, int(ngf * mult / 2)))
        mult = 2 ** (n_downsampling - 2)
        setattr(self, 'up' + str(3), upsample_unit(ngf * mult, int(ngf * mult / 2)))

        output_list = [nn.ReflectionPad2d(3), nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.out_unit = nn.Sequential(*output_list)

    def forward(self, input, p_musk):
        x_init = self.init_conv_unit(torch.cat((input, p_musk), 1))
        down_unit0 = getattr(self, 'down0');
        d0 = down_unit0(x_init)
        down_unit1 = getattr(self, 'down1');
        d1 = down_unit1(d0)
        down_unit2 = getattr(self, 'down2');
        d2 = down_unit2(d1)
        down_unit3 = getattr(self, 'down3');
        d3 = down_unit3(d2)

        p_musk = F.relu(p_musk)
        for i in range(p_musk.shape[0]):
            p_musk[i] /= torch.max(p_musk[i])
        neg_p_musk = 1 - p_musk
        P_input = self.P_init_ReflectionPad(input)
        neg_p_musk = self.P_init_ReflectionPad(neg_p_musk)
        P_x_init, P_musk_init = self.Partail_init_conv(P_input, neg_p_musk)
        P_x_init = self.P_init_norm(P_x_init)
        P_x_init = self.P_init_activarion(P_x_init)
        P_down_unit0 = getattr(self, 'P_down0');
        P_d0, P_musk_d0 = P_down_unit0(P_x_init, P_musk_init)
        P_down_unit1 = getattr(self, 'P_down1');
        P_d1, P_musk_d1 = P_down_unit1(P_d0, P_musk_d0)
        P_down_unit2 = getattr(self, 'P_down2');
        P_d2, P_musk_d2 = P_down_unit2(P_d1, P_musk_d1)
        P_down_unit3 = getattr(self, 'P_down3');
        P_d3, P_musk_d3 = P_down_unit3(P_d2, P_musk_d2)

        d3 = torch.cat((d3, P_d3), 1)

        res = self.resblock_seq(d3)
        # print('resnet size:', res.shape, '\n')  # if input is 256*256 , then resnet output is 16*16

        up_uint0 = getattr(self, 'up0');
        up0 = up_uint0(res)
        up0 = torch.cat((up0, d2), 1)
        up0 = torch.cat((up0, P_d2), 1)
        up_uint1 = getattr(self, 'up1');
        up1 = up_uint1(up0);
        up1 = torch.cat((up1, d1), 1)
        up1 = torch.cat((up1, P_d1), 1)
        up_uint2 = getattr(self, 'up2');
        up2 = up_uint2(up1);

        up_uint3 = getattr(self, 'up3');
        up3 = up_uint3(up2)

        out = self.out_unit(up3)
        return out


def create_PartialConvNet():
    generator = PCNet(input_nc=3, output_nc=3, ngf=16, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d,
                      padding_type='reflect')

    if torch.cuda.is_available():
        generator = generator.cuda()

    generator.apply(weights_init_normal)
    # print_network(generator)

    return generator


############################################################################################
############################################################################################
############################################################################################
############################################################################################
class Gatedconv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gatedconv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeconv2dWithActivation(nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeconv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                         dilation, groups, bias)
        self.mask_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                              dilation, groups, bias)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class Gate_ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(Gate_ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [Gatedconv2dWithActivation(dim, dim, kernel_size=3, padding=p)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Gate_downsample_unit(nn.Module):
    def __init__(self, indim, outdim):
        super(Gate_downsample_unit, self).__init__()
        downsample_list = [
            Gatedconv2dWithActivation(in_channels=indim, out_channels=outdim, kernel_size=3, stride=2, padding=1)]
        self.model = nn.Sequential(*downsample_list)

    def forward(self, x):
        return self.model(x)


class Gate_upsample_unit(nn.Module):
    def __init__(self, indim, outdim):
        super(Gate_upsample_unit, self).__init__()
        upsample_list = [
            GatedDeconv2dWithActivation(in_channels=indim, out_channels=outdim, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)]
        self.model = nn.Sequential(*upsample_list)

    def forward(self, x):
        x = self.model(x)
        return x

class Gate_init_unit(nn.Module):
    def __init__(self,input_nc, ngf=32, kernel_size=7, padding=0):
        super(Gate_init_unit,self).__init__()
        self.Reflectionpad = nn.ReflectionPad2d(3)
        self.GateConv = Gatedconv2dWithActivation(input_nc, ngf, kernel_size=kernel_size, padding=padding)
    def forward(self, input):
        x = self.Reflectionpad(input)
        x = self.GateConv(x)
        return x
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
class Refine_Block(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_downsampling=4, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect',activation = nn.LeakyReLU(0.2, True), requires_grad= True):
        super(Refine_Block, self).__init__()
        init_unit = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                     nn.ReLU(True)]
        self.init_conv_unit = nn.Sequential(*init_unit)
        for i in range(n_downsampling):
            mult = 2 ** i
            setattr(self, 'down' + str(i), downsample_unit(ngf * mult, ngf * mult * 2))
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if(i==0 or i == n_downsampling-1):
                setattr(self, 'up'+str(i), upsample_unit(ngf * mult,  int(ngf * mult / 2)) )
            else:
                setattr(self, 'up'+str(i), upsample_unit(ngf * mult*2,  int(ngf * mult / 2)) )
        output_list = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.out_unit = nn.Sequential(*output_list)

        self.tiny_resnet = nn.Sequential(nn.Conv2d(input_nc, 32,3,1,1),ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer), ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer), nn.Conv2d(32,output_nc,3,1,1))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        x_init = self.init_conv_unit(input)
        down_unit0 = getattr(self, 'down0');
        d0 = down_unit0(x_init)
        down_unit1 = getattr(self, 'down1');
        d1 = down_unit1(d0)
        down_unit2 = getattr(self, 'down2');
        d2 = down_unit2(d1)
        down_unit3 = getattr(self, 'down3');
        d3 = down_unit3(d2)
        up_uint0 = getattr(self, 'up0');
        up0 = up_uint0(d3)
        up0 = torch.cat((up0, d2), 1)
        up_uint1 = getattr(self, 'up1');
        up1 = up_uint1(up0);
        up1 = torch.cat((up1, d1), 1)
        up_uint2 = getattr(self, 'up2');
        up2 = up_uint2(up1);
        up_uint3 = getattr(self, 'up3');
        up3 = up_uint3(up2)
        out0 = self.out_unit(up3)
        out1 = self.tiny_resnet(input)
        return out0+out1


class Refine_Net(nn.Module):
    def __init__(self,input_nc, output_nc, n_blocks=1):
        super(Refine_Net, self).__init__()
        self.n_blocks = n_blocks
        for i in range(n_blocks):
            setattr(self, 'Block' + str(i), Refine_Block(input_nc, output_nc))
    def forward(self, input):
        output_list = []
        for i in range(self.n_blocks):
            block_i = getattr(self, 'Block'+str(i))
            x = block_i(input)
            input = x + input
            output_list.append(x)
        return output_list

def create_refine_nets():
    generator = Refine_Net(input_nc=3, output_nc=3, n_blocks=2)

    if torch.cuda.is_available():
        generator = generator.cuda()

    generator.apply(weights_init_normal)
    # print_network(generator)

    return generator

def create_coarse_nets():
    generator = Derain_GlobalGenerator(input_nc=4, output_nc=3, ngf=16, n_downsampling=4, n_blocks=9,
                                       norm_layer=nn.BatchNorm2d,
                                       padding_type='reflect', requires_grad=False)
    if torch.cuda.is_available():
        generator = generator.cuda()
    generator.apply(weights_init_normal)
    return generator
