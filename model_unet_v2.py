import torch
from pytorch_lightning import Trainer
from torch import nn
from torchmetrics import Accuracy

from dataset import GlassData


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, l1, l2):
        x = torch.cat((l1, l2), 1)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: bool = True,
                 dropout: bool = False,
                 conv_mode: str = 'same'):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.dropout = dropout
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = activation

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                               bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                               bias=True)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        if self.normalization:
            self.norm1 = nn.BatchNorm2d(self.out_channels)
            self.norm2 = nn.BatchNorm2d(self.out_channels)

        if self.dropout:
            self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        if self.dropout:
            y = self.drop(y)

        y = self.conv2(y)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)

        before_pooling = y

        if self.pooling:
            y = self.pool(y)

        return y, before_pooling


class UpBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: bool = True,
                 dropout: bool = False,
                 conv_mode: str = 'same'):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.activation = activation
        self.dropout = dropout
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0

        self.up = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2)

        self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                               bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                               bias=True)

        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        if self.normalization:
            self.norm0 = nn.BatchNorm2d(self.out_channels)
            self.norm1 = nn.BatchNorm2d(self.out_channels)
            self.norm2 = nn.BatchNorm2d(self.out_channels)

        if self.dropout:
            self.drop = nn.Dropout2d(0.1)

        self.concat = Concatenate()

    def forward(self, x, encoder_layer):
        y = self.up(x)
        y = self.act0(y)
        if self.normalization:
            y = self.norm0(y)

        y = self.concat(y, encoder_layer)

        y = self.conv1(y)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        if self.dropout:
            y = self.drop(y)

        y = self.conv2(y)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)

        return y


class Unet(nn.Module):
    def __init__(self, input_channels: int = 1, out_channels: int = 1, use_drop_out: bool = False):
        super(Unet, self).__init__()

        self.down_blocks = [
            DownBlock(in_channels=input_channels, out_channels=32, pooling=True, activation='leaky', normalization=True,
                      conv_mode='same', dropout=use_drop_out),
            DownBlock(in_channels=32, out_channels=64, pooling=True, activation='leaky', normalization=True,
                      conv_mode='same', dropout=use_drop_out),
            DownBlock(in_channels=64, out_channels=128, pooling=True, activation='leaky', normalization=True,
                      conv_mode='same', dropout=use_drop_out),
            DownBlock(in_channels=128, out_channels=256, pooling=True, activation='leaky', normalization=True,
                      conv_mode='same', dropout=use_drop_out),
            DownBlock(in_channels=256, out_channels=512, pooling=True, activation='leaky', normalization=True,
                      conv_mode='same', dropout=use_drop_out),
            DownBlock(in_channels=512, out_channels=1024, pooling=False, activation='leaky', normalization=True,
                      conv_mode='same', dropout=use_drop_out),
        ]

        self.up_blocks = [
            UpBlock(in_channels=1024, out_channels=512, activation='leaky', normalization=True, conv_mode='same',
                    dropout=use_drop_out),
            UpBlock(in_channels=512, out_channels=256, activation='leaky', normalization=True, conv_mode='same',
                    dropout=use_drop_out),
            UpBlock(in_channels=256, out_channels=128, activation='leaky', normalization=True, conv_mode='same',
                    dropout=use_drop_out),
            UpBlock(in_channels=128, out_channels=64, activation='leaky', normalization=True, conv_mode='same',
                    dropout=use_drop_out),
            UpBlock(in_channels=64, out_channels=32, activation='leaky', normalization=True, conv_mode='same',
                    dropout=use_drop_out),
        ]

        self.final_conv = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True)

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # choose sigmoid or softmax output depending on output
        if out_channels > 1:
            # This has to be validated, dimension 1 should be always the "class" dimension
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = nn.Sigmoid()

        self.initialize_params()

    def initialize_params(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        encoder_outputs = []

        for down_block in self.down_blocks:
            x, before_pooling = down_block(x)
            encoder_outputs.append(before_pooling)

        n = len(self.down_blocks)-1

        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, encoder_outputs[-(i + 2)])

        x = self.final_conv(x)
        x = self.output_activation(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = Unet()
    summary(model, input_size=(1, 3, 256, 256))
