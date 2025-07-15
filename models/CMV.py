# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(root_path))
from models.DSConv.DSConv import DCN_Conv

dropout_rate = 0.4

class Attention3D(nn.Module):
    def __init__(self, in_channels):
        super(Attention3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention_weights = torch.sigmoid(attention)
        attention_weights = attention_weights.expand_as(x)
        return x * attention_weights

class FuseImages3D(nn.Module):
    def __init__(self, in_channels):
        super(FuseImages3D, self).__init__()
        self.attention_module = Attention3D(in_channels)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, img1, img2, img3):
        weighted_img1 = self.attention_module(img1)
        weighted_img2 = self.attention_module(img2)
        weighted_img3 = self.attention_module(img3)
        fused_image = weighted_img1 + weighted_img2 + weighted_img3
        fused_image = self.dropout(fused_image)
        return fused_image

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x

class CMV(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch, size=32, device='cuda', kernel_size=3, extend_scope=1.0,
                 if_offset=True):
        super(CMV, self).__init__()
        
        self.n_classes = n_classes
        self.in_ch = in_ch
        self.size = size
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.device = device

        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc0_0 = EncoderConv(self.in_ch, self.size)
        self.enc0_x = DCN_Conv(self.in_ch, self.size, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.enc0_y = DCN_Conv(self.in_ch, self.size, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.enc0_z = DCN_Conv(self.in_ch, self.size, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.enc0_ = self.conv3Dblock(2 * size, size, groups=1)
        self.ec0 = self.conv3Dblock(self.in_ch, size, groups=1)

        self.fuse_images_layer = FuseImages3D(in_channels=self.size)

        self.ec1 = self.conv3Dblock(size, size*2, kernel_size=3, padding=1, groups=1)
        self.ec2 = self.conv3Dblock(size*2, size*2, groups=1)
        self.ec3 = self.conv3Dblock(size*2, size*4, groups=1)
        self.ec4 = self.conv3Dblock(size*4, size*4, groups=1)
        self.ec5 = self.conv3Dblock(size*4, size*8, groups=1)
        self.ec6 = self.conv3Dblock(size*8, size*8, groups=1)
        self.ec7 = self.conv3Dblock(size*8, size*16, groups=1)

        self.dc9 = nn.ConvTranspose3d(size*16+1, size*16, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(size*8 + size*16, size*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(size*8, size*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(size*8, size*8, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(size*4 + size*8, size*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(size*4, size*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(size*4, size*4, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(size*2 + size*4, size*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(size*2, size*2, kernel_size=3, stride=1, padding=1)

        self.final = nn.ConvTranspose3d(self.size * 2, n_classes, kernel_size=3, padding=1, stride=1)
        self.to(device)
        initialize_weights(self)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), groups=1,
                    padding_mode='replicate'):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
                      groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, emb_codes):

        x1 = x[:, 0:1, :, :, :]  # CBCT
        x2 = x[:, 1:2, :, :, :]  # Line_Mask

        x_x = self.enc0_x(x)
        x_y = self.enc0_y(x)
        x_z = self.enc0_z(x)
        x_xyz = self.fuse_images_layer(x_x, x_y, x_z)
        h = self.ec0(x)
        h = self.enc0_(cat([x_xyz, h], dim=1))
        feat_0 = self.ec1(h)

        h = self.pool0(feat_0)
        h = self.ec2(h)
        feat_1 = self.ec3(h)

        h = self.pool1(feat_1)
        h = self.ec4(h)
        feat_2 = self.ec5(h)

        h = self.pool2(feat_2)
        h = self.ec6(h)
        h = self.ec7(h)

        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        h = torch.cat((h, emb_pos), dim=1)

        h = torch.cat((self.dc9(h), feat_2), dim=1)

        h = self.dc8(h)
        h = self.dc7(h)

        h = torch.cat((self.dc6(h), feat_1), dim=1)
        h = self.dc5(h)
        h = self.dc4(h)

        h = torch.cat((self.dc3(h), feat_0), dim=1)
        h = self.dc2(h)
        h = self.dc1(h)

        h = self.final(h)

        return torch.sigmoid(h)

