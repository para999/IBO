import time
import torch
import torch.nn as nn
from model.IBO import common
from model.IBO.DRAT import DRAT


class Encoder_LR(nn.Module):
    def __init__(self, feats=64, scale=4):
        super().__init__()
        self.scale = scale
        self.E = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, lr):
        lr_ave = self.E(lr).squeeze(-1).squeeze(-1)
        ldp = self.mlp(lr_ave)
        return ldp


class Encoder_SR(nn.Module):
    def __init__(self, feats=64, scale=4):
        super().__init__()
        if scale == 2:
            in_dim = 15
        elif scale == 3:
            in_dim = 30
        elif scale == 4:
            in_dim = 51
        else:
            in_dim = 0
            print('Upscale error!!!!')

        self.LR = nn.Sequential(
            nn.Conv2d(in_dim, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.SR = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=7, stride=7, padding=0),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats * 4 * 2, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, lr, sr):
        sr0 = self.pixel_unshuffle(sr)
        lr = torch.cat([lr, sr0], dim=1)
        lr_ave = self.LR(lr).squeeze(-1).squeeze(-1)
        sr_ave = self.SR(sr).squeeze(-1).squeeze(-1)
        sdp = self.mlp(torch.cat([lr_ave, sr_ave], dim=1))
        return sdp


class IBO(nn.Module):
    def __init__(self, scale):
        super(IBO, self).__init__()
        self.Restorer_Upper_Level = DRAT(upscale=scale, IBO='upper')
        self.Restorer_Lower_Level = DRAT(upscale=scale, IBO='lower')
        self.Estimator_LR = Encoder_LR(scale=scale)
        self.Estimator_SR = Encoder_SR(scale=scale)

    # def freeze_module(self, module):
    #     for param in module.parameters():
    #         param.requires_grad = False

    def forward(self, lr, hr=None, sr=True):
        if hr is None:
            # Upper Level Optimization
            ldp = self.Estimator_LR(lr)
            sr_upper_level = self.Restorer_Upper_Level(lr, ldp)
            if not sr:
                return sr_upper_level
            # Iterative Bi-level Optimization
            sdp = self.Estimator_SR(lr, sr_upper_level.detach())
            sr_lower_level = self.Restorer_Lower_Level(lr, sdp)
            return sr_lower_level
        else:
            # Lower-Level Optimization
            sdp = self.Estimator_SR(lr, hr)
            sr_lower_level = self.Restorer_Lower_Level(lr, sdp)
            return sr_lower_level


if __name__ == '__main__':
    scale = 4
    height = 48
    width = 48
    batch_size = 1
    lr = torch.randn((batch_size, 3, height, width))
    hr = torch.randn((batch_size, 3, height * scale, width * scale))
    print("height", height, "width", width)
    model = IBO(scale=scale)
    model.eval()
    SR = True
    HR = None
    start_time = time.time()
    if SR is False and HR is None:
        sr = model(lr, None, SR)
        print('sr shape', sr.shape)
    elif SR is False and HR is not None:
        sr = model(lr, hr, SR)
        print('sr shape', sr.shape)
    elif SR is True and HR is None:
        sr = model(lr, None, SR)
        print('sr shape', sr.shape)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution Time: {execution_time:.6f} seconds')