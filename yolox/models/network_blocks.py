import torch
import torch.nn as nn
import pytorch_lightning as pl

def get_activation(name="silu", inplace=True):
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(inplace=inplace)
    else:
        raise f"ValueError: Activation function only support [silu, relu, lrelu] for [nn.Silu, nn.RelU, " \
              f"nn.LeakyRelU] respectively, but got {name} "


class BaseConv(pl.LightningModule):
    """
    spatial resolution non-degenerated convolution when stride is 1, so the padding should meet:
    padding = (kernel_size - 1) // 2. (Note: kernel_size is odd)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act="silu"):
        super(BaseConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.act = get_activation(act, inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(pl.LightningModule):
    """
    BaseConv + 1x1 Conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act="silu"):
        super().__init__()
        self.depthconv = BaseConv(in_channels,
                                  in_channels,
                                  kernel_size,
                                  stride,
                                  groups=in_channels,
                                  act=act)
        self.pointconv = BaseConv(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  act=act)

    def forward(self, x):
        return self.pointconv(self.depthconv(x))


class ResLayer(pl.LightningModule):
    """
    Residual layer which keep the same size of channel before and after Conv
    """

    def __init__(self, in_channels):
        super(ResLayer, self).__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels,
                               mid_channels,
                               kernel_size=1,
                               stride=1,
                               act="lrelu"
                               )
        self.layer2 = BaseConv(mid_channels,
                               in_channels,
                               kernel_size=3,
                               stride=1,
                               act="lrelu")

    def forward(self, x):
        y = self.layer2(self.layer1(x))
        return x + y


class Bottleneck(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act="silu"):
        super().__init__()
        hidden_channels = in_channels * expansion
        Conv = DWConv if depthwise else BaseConv
        self.layer1 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.layer2 = Conv(hidden_channels, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.layer2(self.layer1(x))
        if self.use_add:
            y = x + y
        return y


class SPPBottleNeck(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 9, 13), act="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.bottleneck = pl.LightningModuleList(
            [nn.MaxPool2d(ks, padding=(ks // 2)) for ks in kernel_size]
        )
        cat_channels = hidden_channels * (len(kernel_size) + 1)
        self.conv2 = BaseConv(cat_channels, out_channels, kernel_size=1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [b(x) for b in self.bottleneck], dim=1)
        return self.conv2(x)


class CSPLayer(pl.LightningModule):
    """C3 in yolox, CSP Bottleneck with 3 convolutions"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 n=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act="silu"):
        """
        :param n: number of bottlenecks
        :param shortcut: whether to use residual connection in BottleNeck
        :param expansion: = hidden_channels // in_channels
        :param depthwise: whether to use DWConv or BaseConv
        """
        super(CSPLayer, self).__init__()
        hidden_channels = int(in_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, kernel_size=1, stride=1, act=act)

        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, expansion, depthwise, act)
            for _ in range(n)
        ]
        self.bottleneck = nn.Sequential(*module_list)

    def forward(self, x):
        y_1 = self.conv1(x)
        y_2 = self.conv2(x)
        y_1 = self.bottleneck(y_1)
        y = torch.cat((y_1, y_2), dim=1)
        return self.conv3(y)


class Fcous(pl.LightningModule):
    """
    Blocksize fixes to 4.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        super(Fcous, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, kernel_size, stride, act)

    def forward(self, x):
        x_1 = x[..., ::2, ::2]
        x_2 = x[..., 1::2, ::2]
        x_3 = x[..., ::2, 1::2]
        x_4 = x[..., 1::2, 1::2]
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        return self.conv(x)
