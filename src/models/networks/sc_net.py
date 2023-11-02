import torch
from typing import Sequence, Tuple, Union
import torch.nn as nn
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from src.models.networks.module import ResEncoder
import torch.nn.functional as F
class Conv3d_wd(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super(Conv3dBlock, self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                    bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes,  norm_cfg, activation_cfg,kernel_size=3, stride=1, padding=1,
                                    bias=False, weight_std=weight_std)
        self.transconv = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=1, stride=1, bias=False,
                                     weight_std=weight_std)

    def forward(self, x):
        residual = x
        out = self.resconv1(x)
        out = self.resconv2(out)
        if out.shape[1] != residual.shape[1]:
            residual = self.transconv(residual)
        out = out + residual
        return out
<<<<<<< HEAD

=======
>>>>>>> 7b61754c3e71aee6f37a3a00232a1dc679f066b2
class SC_Net(nn.Module):
    def __init__(self,
        in_channels: 384,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        depths: Sequence[int] = (2, 2, 2, 2),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 8,
        #num_classes: int = 16,
        conv_op=nn.Conv3d,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        deep_supervision=False,
        norm_cfg='BN', activation_cfg='ReLU', weight_std=False,

    ):
            #>>> net =  UnetrBasicBlock(in_channels = 384, out_channels = , conv_op=conv_op, norm_name=norm_name, res_block=res_block)
        super().__init__()
        self.num_layers = 8
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(2, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.upscale_logits_ops = []
        self.img_size = img_size
        self.upscale_logits_ops.append(lambda x: x)
        self.final = []
        self.final = nn.ModuleList(self.final)
        #self.num_classes = num_classes
        self.conv_op = conv_op
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.transposeconv_stage2 = nn.ConvTranspose3d(512,256,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(256,128,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(128,64,kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.transposeconv_skip3 = nn.ConvTranspose3d(768,512,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_skip2 =  nn.ConvTranspose3d(768,512,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_skip1_1 =  nn.ConvTranspose3d(768,512,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_skip1_2 =  nn.ConvTranspose3d(512,256,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_skip0_1 =  nn.ConvTranspose3d(768,512,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_skip0_2 =  nn.ConvTranspose3d(512,256,kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_skip0_3 =  nn.ConvTranspose3d(256,128,kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.stage2_de = ResBlock(1408,512,norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(896,256,norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(448,128,norm_cfg, activation_cfg, weight_std=weight_std)

        self.cls_conv = nn.Conv3d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.resencoder = ResEncoder(depth=7)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

<<<<<<< HEAD
    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, res_encoder_output):
        #res_encoder_output = self.resencoder(x)
        transencoder_output, hidden_states_out = self.vit(res_encoder_output[2])
        skip3 = self.transposeconv_skip3(self.proj_feat(transencoder_output))
        skip2 = self.transposeconv_skip2(self.proj_feat(hidden_states_out[-3]))
        x = torch.cat((res_encoder_output[2], skip3, skip2), dim=1)
        x = self.stage2_de(x)
        x = self.transposeconv_stage2(x)
        skip1 = self.transposeconv_skip1_1(self.proj_feat(hidden_states_out[-5]))
        skip1 = self.transposeconv_skip1_2(skip1)
        x = torch.cat((res_encoder_output[1], x, skip1),dim=1)
        x = self.stage1_de(x)
        x = self.transposeconv_stage1(x)
        skip0 = self.transposeconv_skip0_1(self.proj_feat(hidden_states_out[-7]))
        skip0 = self.transposeconv_skip0_2(skip0)
        skip0 = self.transposeconv_skip0_3(skip0)
        x = torch.cat((res_encoder_output[0], x, skip0),dim=1)
        x = self.stage0_de(x)
        x = self.transposeconv_stage0(x)
        output = self.cls_conv(x)
        output = self.sigmoid(output)

=======
    def forward(self, res_encoder_output):
        #res_encoder_output = self.resencoder(x)
        transencoder_output,hidden_states_out = self.vit(res_encoder_output[2])
        skip2 = self.transposeconv_skip2(hidden_states_out[-3])
        x = torch.cat(res_encoder_output[2],x,skip2,dim=1)
        x = self.stage2_de(x)
        x = self.transposeconv_stage2(x)
        skip1 = self.transposeconv_skip1_1(hidden_states_out[-5])
        skip1 = self.transposeconv_skip1_2(skip1)
        x = torch.cat(res_encoder_output[1],x,skip1,dim=1)
        x = self.stage1_de(x)
        x = self.transposeconv_stage1(x)
        skip0 = self.transposeconv_skip0_1(hidden_states_out[-7])
        skip0 = self.transposeconv_skip0_2(skip0)
        skip0 = self.transposeconv_skip0_3(skip0)
        x = torch.cat(res_encoder_output[0],x,skip0,dim=1)
        x = self.stage0_de(x)
        x = self.transposeconv_stage0(x)
        output = self.cls_conv(x)
        output = nn.Sigmoid(output)

>>>>>>> 7b61754c3e71aee6f37a3a00232a1dc679f066b2
        return output




#class ScNet(nn.Module):
#    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False):

