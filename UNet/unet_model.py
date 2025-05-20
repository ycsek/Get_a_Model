"""Full assembly of the parts to form the Camouflager network for model hijacking"""

from .unet_parts import *
import torch
import torch.nn as nn


class UNetEncoder(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNetEncoder, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, [x4, x3, x2, x1]


class UNetDecoder(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(UNetDecoder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, skip_o, skip_h):
        # Combine skip connections from both encoders by addition
        skip = [torch.add(s_o, s_h) for s_o, s_h in zip(skip_o, skip_h)]
        x = self.up1(x, skip[0])
        x = self.up2(x, skip[1])
        x = self.up3(x, skip[2])
        x = self.up4(x, skip[3])
        logits = self.outc(x)
        return logits


class Camouflager(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Camouflager, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Two encoders for hijackee (x_o) and hijacking (x_h) samples
        self.encoder_o = UNetEncoder(n_channels, bilinear)
        self.encoder_h = UNetEncoder(n_channels, bilinear)
        # One decoder to generate camouflaged samples
        self.decoder = UNetDecoder(n_classes, bilinear)

    def forward(self, x_o, x_h):
        # Encode hijackee and hijacking samples
        x5_o, skip_o = self.encoder_o(x_o)
        x5_h, skip_h = self.encoder_h(x_h)
        # Combine bottleneck features by addition
        x5 = torch.add(x5_o, x5_h)
        # Decode to produce camouflaged sample
        logits = self.decoder(x5, skip_o, skip_h)
        return logits

    def use_checkpointing(self):
        self.encoder_o.inc = torch.utils.checkpoint(self.encoder_o.inc)
        self.encoder_o.down1 = torch.utils.checkpoint(self.encoder_o.down1)
        self.encoder_o.down2 = torch.utils.checkpoint(self.encoder_o.down2)
        self.encoder_o.down3 = torch.utils.checkpoint(self.encoder_o.down3)
        self.encoder_o.down4 = torch.utils.checkpoint(self.encoder_o.down4)
        self.encoder_h.inc = torch.utils.checkpoint(self.encoder_h.inc)
        self.encoder_h.down1 = torch.utils.checkpoint(self.encoder_h.down1)
        self.encoder_h.down2 = torch.utils.checkpoint(self.encoder_h.down2)
        self.encoder_h.down3 = torch.utils.checkpoint(self.encoder_h.down3)
        self.encoder_h.down4 = torch.utils.checkpoint(self.encoder_h.down4)
        self.decoder.up1 = torch.utils.checkpoint(self.decoder.up1)
        self.decoder.up2 = torch.utils.checkpoint(self.decoder.up2)
        self.decoder.up3 = torch.utils.checkpoint(self.decoder.up3)
        self.decoder.up4 = torch.utils.checkpoint(self.decoder.up4)
        self.decoder.outc = torch.utils.checkpoint(self.decoder.outc)
