import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2)-x.size(2), skip.size(3)-x.size(3)
        x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], dim=1))

class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base); self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base, base*2); self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(base*2, base*4); self.p3 = nn.MaxPool2d(2)
        self.b  = ConvBlock(base*4, base*8)
        self.u3 = UpBlock(base*8, base*4)
        self.u2 = UpBlock(base*4, base*2)
        self.u1 = UpBlock(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def encode(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b  = self.b(self.p3(e3))
        return [e1, e2, e3, b]

    def forward(self, a, b):
        fa, fb = self.encode(a), self.encode(b)
        x = self.u3(torch.abs(fa[3]-fb[3]), torch.abs(fa[2]-fb[2]))
        x = self.u2(x, torch.abs(fa[1]-fb[1]))
        x = self.u1(x, torch.abs(fa[0]-fb[0]))
        return self.out(x)