import torch
import torch.nn as nn


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding1, padding2):
        super(_Up, self).__init__()
        self._convtrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self._conv = double_conv(in_channels, out_channels, padding1=padding1, padding2=padding2)

    def forward(self, x, y):
        x = self._convtrans(x)
        _, _, h, w = x.shape
        diffY = (y.size()[2] - x.size()[2]) // 2
        diffX = (y.size()[3] - x.size()[3]) // 2
        y = y[:, :, diffY:diffY + x.size()[2], diffX:diffX + x.size()[3]]
        out = torch.cat([x, y], dim=1)
        out = self._conv(out)
        return out


def double_conv(in_ch, out_ch, padding1=0, padding2=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=padding1),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, 3, padding=padding2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


def make_down_layers(channels_min, deep):
    layers = nn.ModuleList()
    channels = channels_min
    for i in range(deep):
        layers.append(nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(channels, channels * 2, padding1=0, padding2=1)
        ))
        channels *= 2
    return layers


def make_up_layers(channels_max, deep):
    layers = nn.ModuleList()
    channels = int(channels_max)
    for i in range(deep):
        layers.append(_Up(channels, int(channels / 2), kernel_size=2, stride=2, padding1=0, padding2=1))
        channels = int(channels / 2)
    return layers



