#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

# 基于CSP结构的Backbone骨干网络

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):                 # 以Module类为父类定义YOLOPAFPN类
    def __init__(
        self,
        depth=1.0,                          # 网络深度因子
        width=1.0,                          # 网络宽度因子
        in_features=("dark3", "dark4", "dark5"),    # 输入特征层，"dark3", "dark4", "dark5"分别指代较浅、中等、较深的层级
        in_channels=[256, 512, 1024],               # 输入的通道数，有三个通道数，分别对应"dark3", "dark4", "dark5"
        depthwise=False,                            # 是否使用深度卷积
        act="silu",                                 # 激活函数silu
    ):
        super().__init__()                          # 父类初始化
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)  # 创建CSPDarknet网络作为主体部分
        self.in_features = in_features              # 创建输入特征属性
        self.in_channels = in_channels              # 创建输入通道数属性
        Conv = DWConv if depthwise else BaseConv    # depthwise为True:深度卷积；为Fulse：不使用深度卷积

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest") # 创建上采样属性，用于特征图尺寸放大
        self.lateral_conv0 = BaseConv(              
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )                                           # 创建侧边卷积层(输入通道数，输出通道数，卷积核大小，卷积核步长，激活函数)
                                                    # 此卷积层的作用是将来自深层的特征图通道数减小为适合于中层的特征图通道数，以进行FPN操作
        self.C3_p4 = CSPLayer(                      
            int(2 * in_channels[1] * width),        # 输入通道数
            int(in_channels[1] * width),            # 输出通道数
            round(3 * depth),                       # 结构重复次数
            False,                                  # 是否使用特殊结构（Bottleneck、Res）
            depthwise=depthwise,                    # 是否使用深度卷积
            act=act,                                # 激活函数类型
        )                                           # 创建CSP层

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )                                           # 创建侧边卷积层(输入通道数，输出通道数，卷积核大小，卷积核步长，激活函数)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )                                       

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
