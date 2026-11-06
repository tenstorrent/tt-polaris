#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
from ttsim.ops.tensor import require_shape_list
import numpy as np

def align_img_dim(img_dim):
    Q = img_dim // Bottleneck.expansion
    R = img_dim % Bottleneck.expansion
    return img_dim + R if R != 0 else img_dim

class Bottleneck(SimNN.Module):
    expansion = 4
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.in_channels  = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.stride       = cfg.get('stride', 1)
        self.downsample   = cfg.get('downsample', None)

        conv_dims = [
                #IC, OC, K, P, S
                (self.in_channels,  self.out_channels, 1, 0, 1),
                (self.out_channels, self.out_channels, 3, 1, self.stride),
                (self.out_channels, self.out_channels*Bottleneck.expansion, 1, 0, 1),
                ]
        oplist = []
        for i, (ic, oc, k, p, s) in enumerate(conv_dims):
            conv = F.Conv2d(self.name + f'.conv{i}', ic, oc, kernel_size=k, padding=p, stride=s)
            bn   = F.BatchNorm2d(self.name + f'.bn{i}', oc)
            oplist += [conv, bn]

        self.op_blk  = F.SimOpHandleList(oplist)
        self.relu    = F.Relu(self.name + f'.relu')

        if self.downsample is not None:
            xi = self.downsample['in_channels']
            xo = self.downsample['out_channels']
            xs = self.downsample['stride']
            self.conv_ds = F.Conv2d(self.name + '.conv_ds', xi, xo, kernel_size=1, padding=0, stride=xs)
            self.bn_ds   = F.BatchNorm2d(self.name + '.bn_ds', xo)

        super().link_op2module()

    def __call__(self, x):
        y = self.op_blk(x)
        if self.downsample is None:
            z = y + x
        else:
            x = self.conv_ds(x)
            x = self.bn_ds(x)
            z = y + x
        w = self.relu(z)
        return w

class ResNet(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name           = name
        self.in_channels    = 64
        self.bs             = cfg.get('bs',             1)
        self.num_channels   = cfg.get('num_channels',   3)
        self.init_stride    = cfg.get('init_stride',    2)
        self.num_classes    = cfg.get('num_classes', 1000)
        self.layers         = cfg.get('layers', [3,4,6,3])
        self.img_height     = align_img_dim(cfg.get('img_height',224))
        self.img_width      = align_img_dim(cfg.get('img_width', 224))
        self.use_adaptive_pool  = cfg.get('use_adaptive_pool', False)

        layers_msg = f"ResNet.layers should be [int, int, int, int] with each member > 0: {self.layers}"
        assert isinstance(self.layers, list) and \
               len(self.layers) == 4 and \
               all([isinstance(x, int) and x > 0 for x in self.layers]), layers_msg

        #ops
        self.conv0    = F.Conv2d(self.name +'.conv0', self.num_channels, self.in_channels, kernel_size=7, stride=self.init_stride, padding=3)
        self.bn0      = F.BatchNorm2d(self.name + '.bn0', self.in_channels)
        self.relu0    = F.Relu(self.name + '.relu0')
        self.maxpool0 = F.MaxPool2d(self.name + '.maxpool0', kernel_size=3, stride=2) #, padding=1)

        self.layer1   = SimNN.ModuleList(self._make_layer(0, self.layers[0], planes= 64, stride=1))
        self.layer2   = SimNN.ModuleList(self._make_layer(1, self.layers[1], planes=128, stride=2))
        self.layer3   = SimNN.ModuleList(self._make_layer(2, self.layers[2], planes=256, stride=2))
        self.layer4   = SimNN.ModuleList(self._make_layer(3, self.layers[3], planes=512, stride=2))

        # For high-resolution images, use adaptive pooling to ensure correct output dimensions
        if self.use_adaptive_pool:
            self.avgpool = F.AveragePool2d(self.name + '.avgpool', output_size=(1, 1), adaptive=True)
        else:
            self.avgpool = F.AveragePool2d(self.name + '.avgpool', kernel_shape=(7, 7))

        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
                'x_in': F._from_shape('x_in', [self.bs, self.num_channels, self.img_height, self.img_width],
                                   is_param=False, np_dtype=np.float32),
                }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self):
        return 0

    def __call__(self):
        x = self.input_tensors['x_in']
        shp = require_shape_list(
            x.shape,
            "basicresnet ResNet.__call__: input x_in tensor shape must be set",
        )
        batch, img_chnl, img_width, img_height = shp

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        for blk in self.layer1: x = blk(x)
        for blk in self.layer2: x = blk(x)
        for blk in self.layer3: x = blk(x)
        for blk in self.layer4: x = blk(x)
        x = self.avgpool(x)

        assert x.shape is not None, "basicresnet ResNet.__call__: pooled tensor shape must be set"
        # int() keeps the dims plain: np.prod returns a numpy integer.
        x_size   = int(np.prod(x.shape[1:])) # don't use batchsize in finding reshape dim
        q_factor = x_size // Bottleneck.expansion
        r_factor = x_size % Bottleneck.expansion
        assert r_factor == 0, f"Input Image Size did not result into a neat multiple of Bottleneck.expansion = {Bottleneck.expansion}"
        reshape_dim = q_factor * Bottleneck.expansion
        x = x.reshape(batch, reshape_dim) #type: ignore

        x = T.matmul(x, F._from_shape(self.name + '.fc0.param', [reshape_dim, self.num_classes], is_param=True))

        return x

    def _make_layer(self, lyr_num, blocks, planes, stride=1):
        downsample_cfg = None
        block_list     = []

        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample_cfg = {
                    'in_channels': self.in_channels,
                    'out_channels': planes * Bottleneck.expansion,
                    'stride': stride
                    }
        bb = Bottleneck(f'layer{lyr_num}.bb0', {'in_channels': self.in_channels,
                                'out_channels': planes,
                                'downsample': downsample_cfg,
                                'stride': stride})
        block_list.append(bb)
        self.in_channels = planes * Bottleneck.expansion

        for i in range(blocks-1):
            block_list.append(Bottleneck(f'layer{lyr_num}.bb{i+1}', {
                'in_channels': self.in_channels,
                'out_channels': planes
                }))

        return block_list

def run_standalone(outdir: str ='.')->None:
    resnet_cfgs = {
            'resnet_50_256x224' : {'layers': [3,4,6,3],  'num_classes': 1000, 'num_channels': 3,
                                   'img_height': 256, 'img_width': 224},
            #'resnet_101_224x224': {'layers': [3,4,23,3], 'num_classes': 10, 'num_channels': 1},
            #'resnet_152_224x224': {'layers': [3,8,36,3], 'num_classes': 10, 'num_channels': 2},
            }

    for k,v in resnet_cfgs.items():
        print(f'Creating ResNet({k})....')
        rn_model = ResNet(k,v)
        rn_model.create_input_tensors()
        y = rn_model()
        print('Input:    ', rn_model.input_tensors['x_in'].shape)
        print('Output:   ', y.shape)
        gg = rn_model.get_forward_graph()
        print('Dumping ONNX...')
        gg.graph2onnx(f'{outdir}/{k}.onnx', do_model_check=True)
        print('-'*40,'\n')


if __name__ == '__main__':
    run_standalone()
