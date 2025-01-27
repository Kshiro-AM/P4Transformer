import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

import pointnet2_utils
from point_4d_convolution import *
from transformer import *

class P4Transformer(nn.Module):
    def __init__(self, radius=0.9, nsamples=3*3, num_classes=12):
        
        super(P4Transformer, self).__init__()
        
        self.conv1 = P4DConv(in_planes=3,
                             mlp_planes=[32,64,128],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.conv2 = P4DConv(in_planes=128,
                             mlp_planes=[128, 128, 256],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.conv3 = P4DConv(in_planes=256,
                             mlp_planes=[256,256,512],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*radius, nsamples],
                             temporal_kernel_size=3,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[1,1])

        self.conv4 = P4DConv(in_planes=512,
                             mlp_planes=[512,512,1024],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*2*radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=2,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.emb_relu = nn.ReLU()
        self.transformer = Transformer(dim=1024, depth=2, heads=4, dim_head=256, mlp_dim=1024)
        
        self.templateConv = TemplateConv(in_planes=3,
                            mlp_planes=[32,64,128],
                            mlp_batch_norm=[True, True, True],
                            mlp_activation=[True, True, True],
                            spatial_kernel_size=[radius, nsamples],
                            temporal_kernel_size=1,
                            spatial_stride=4,
                            temporal_stride=1,
                            temporal_padding=[0,0])

        self.deconv4 = P4DTransConv(in_planes=1024,
                                    mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=512)

        self.deconv3 = P4DTransConv(in_planes=256,
                                    mlp_planes=[256, 256],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=256)

        self.deconv2 = P4DTransConv(in_planes=256,
                                    mlp_planes=[128, 128],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=128)

        self.deconv1 = P4DTransConv(in_planes=128,
                                    mlp_planes=[128, 128],
                                    mlp_batch_norm=[True, True, True],
                                    mlp_activation=[True, True, True],
                                    original_planes=3)

        self.outconv = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0)
        
        self.connectoutconv1 = nn.Conv2d(in_channels=512, out_channels=16384, kernel_size=1, stride=1, padding=0)
        self.connectoutconv2 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0)
        
        self.templateout = nn.Conv2d(in_channels=256, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.templateout2 = nn.Conv2d(in_channels=16384, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.mlp = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, xyzs, rgbs, templates):
        
        device = xyzs.get_device()

        new_xyzs1, new_features1 = self.conv1(xyzs, rgbs)

        new_xyzs2, new_features2 = self.conv2(new_xyzs1, new_features1)

        new_xyzs3, new_features3 = self.conv3(new_xyzs2, new_features2)

        new_xyzs4, new_features4 = self.conv4(new_xyzs3, new_features3)

        B, L, _, N = new_features4.size()


        features = new_features4.permute(0, 1, 3, 2)                                                                                        # [B, L, n2, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n2, C]

        embedding = self.emb_relu(features)

        features = self.transformer(embedding)

        features = torch.reshape(input=features, shape=(B, L, N, features.shape[2]))                                                        # [B, L, n2, C]
        features = features.permute(0, 1, 3, 2)
        
        new_features4 = features
        new_xyzsd4, new_featuresd4 = self.deconv4(new_xyzs4, new_xyzs3, new_features4, new_features3)

        new_xyzsd3, new_featuresd3 = self.deconv3(new_xyzsd4, new_xyzs2, new_featuresd4, new_features2)

        new_xyzsd2, new_featuresd2 = self.deconv2(new_xyzsd3, new_xyzs1, new_featuresd3, new_features1)

        new_xyzsd1, new_featuresd1 = self.deconv1(new_xyzsd2, xyzs, new_featuresd2, rgbs)

        out = self.outconv(new_featuresd1.transpose(1,2)).transpose(1,2)
        
        # TemplateNet
        
        template_features = torch.zeros_like(templates, device=device)
        template_features = template_features.permute(0, 2, 1)
        new_template_xyzs, new_tempalte_features = self.templateConv(templates, template_features)
        
        new_featuresd1 = new_featuresd1.detach()
        new_featuresd = self.templateout2(new_featuresd1.transpose(1, 3)).transpose(1, 3)
        
        new_tempalte_features_expanded = new_tempalte_features.expand(new_featuresd.size()[0], 3, -1, -1)
        
        template_conv_in = torch.cat((new_featuresd, new_tempalte_features_expanded), dim=2)
        template_mlp_out = self.templateout(template_conv_in.transpose(1,2)).transpose(1,2)
        
        template_out = self.mlp(template_mlp_out).squeeze().permute(0, 2, 1)
        
        # conncet out
        
        final_out = self.connectoutconv1(template_conv_in.transpose(1,3)).transpose(1,3)
        final_out = self.connectoutconv2(final_out.transpose(1,2)).transpose(1,2)

        return out, template_out, final_out

