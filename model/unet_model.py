from model.unet_parts import *
import torch.nn as nn
import math


class Unet(nn.Module):
    def __init__(self, target, data_dict, in_channels, out_channels, init_feature, deep, input_size=(640, 540)):
        super(Unet, self).__init__()
        self.target = target
        self.data_dict = data_dict
        self.input_size, self.gap = self.calculate_size_gap(input_size, deep)

        self._inc = double_conv(in_channels, init_feature)
        self._down_layers = make_down_layers(init_feature, deep)
        self._up_layers = make_up_layers(init_feature * math.pow(2, deep), deep)
        # classifier
        self._out = nn.Sequential(
            nn.Conv2d(init_feature, init_feature, 3),
            nn.Conv2d(init_feature, out_channels, 3),
            nn.Sigmoid()
        )

    def forward(self, imgs):
        c = [self._inc(imgs)]
        for i in range(len(self._down_layers)):
            c.append(self._down_layers[i](c[i]))

        out = c[len(c) - 1]
        for i in range(len(self._up_layers)):
            j = len(self._up_layers) - i
            out = self._up_layers[i](out, c[j - 1])
        out = self._out(out)
        return out

