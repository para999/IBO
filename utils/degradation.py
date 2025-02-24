import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# Blur
def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma


def isotropic_gaussian_kernel(kernel_size, sigma):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1, 2], keepdim=True)


def anisotropic_gaussian_kernel(kernel_size, covar):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
    xy = torch.stack([xx, yy], -1).view(1, -1, 2)

    covar = covar.cpu()
    inverse_sigma = torch.inverse(covar).cuda()
    # inverse_sigma = np.linalg.inv(covar)
    # inverse_sigma = torch.tensor(inverse_sigma).cuda()
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(1, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def random_isotropic_gaussian_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0):
    x = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    kernel = isotropic_gaussian_kernel(kernel_size, x)
    return kernel


def stable_isotropic_gaussian_kernel(kernel_size=21, sigma=4.0):
    x = torch.ones(1).cuda() * sigma
    kernel = isotropic_gaussian_kernel(kernel_size, x)
    return kernel


def random_anisotropic_gaussian_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0):
    theta = torch.rand(1).cuda() * math.pi
    sigma_x = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    sigma_y = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    covar = cal_sigma(sigma_x, sigma_y, theta)
    kernel = anisotropic_gaussian_kernel(kernel_size, covar)
    return kernel


def stable_anisotropic_gaussian_kernel(kernel_size=21, theta=0, sigma_x=0.2, sigma_y=4.0):
    theta = torch.ones(1).cuda() * theta / 180 * math.pi
    sigma_x = torch.ones(1).cuda() * sigma_x
    sigma_y = torch.ones(1).cuda() * sigma_y
    # print(sigma_x, sigma_y)
    covar = cal_sigma(sigma_x, sigma_y, theta)
    kernel = anisotropic_gaussian_kernel(kernel_size, covar)
    return kernel


# total random gaussian kernel
def generate_random_kernel(kernel_size=21, blur_type='iso_gaussian', sigma_min=0.2, sigma_max=4.0):
    if blur_type == 'iso_gaussian':
        return random_isotropic_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max)
    elif blur_type == 'aniso_gaussian':
        return random_anisotropic_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max)


def generate_stable_kernel(kernel_size=21, blur_type='iso_gaussian', sigma=1.8, theta=0, sigma_x=0.2, sigma_y=4.0):
    if blur_type == 'iso_gaussian':
        return stable_isotropic_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    elif blur_type == 'aniso_gaussian':
        return stable_anisotropic_gaussian_kernel(kernel_size=kernel_size, theta=theta, sigma_x=sigma_x,
                                                  sigma_y=sigma_y)


class BatchRandonKernel:
    def __init__(self, blur_type_list, blur_type_prob, kernel_size=21, sigma_min=0.2, sigma_max=4.0):
        super(BatchRandonKernel, self).__init__()
        self.kernel_size = kernel_size
        self.blur_type_list = blur_type_list
        self.blur_type_prob = blur_type_prob

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, batch):
        batch_kernel = torch.zeros(batch, self.kernel_size, self.kernel_size).float().cuda()
        for i in range(batch):
            blur_type = random.choices(self.blur_type_list, self.blur_type_prob)[0]
            # print("blur_type", blur_type)
            kernel_i = generate_random_kernel(kernel_size=self.kernel_size, blur_type=blur_type,
                                              sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                              )
            batch_kernel[i, :, ...] = kernel_i
        return batch_kernel


class BatchStableKernel:
    def __init__(self, blur_type_list, blur_type_prob, kernel_size=21, sigma=1.8, theta=0, sigma_x=0.2, sigma_y=4.0):
        super(BatchStableKernel, self).__init__()
        self.kernel_size = kernel_size
        self.blur_type_list = blur_type_list
        self.blur_type_prob = blur_type_prob

        self.sigma = sigma
        self.theta = theta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, batch):
        batch_kernel = torch.zeros(batch, self.kernel_size, self.kernel_size).float().cuda()
        for i in range(batch):
            blur_type = random.choices(self.blur_type_list, self.blur_type_prob)[0]
            # print("blur_type", blur_type)
            kernel_i = generate_stable_kernel(kernel_size=self.kernel_size, blur_type=blur_type, sigma=self.sigma,
                                              theta=self.theta, sigma_x=self.sigma_x, sigma_y=self.sigma_y)
            batch_kernel[i, :, ...] = kernel_i
        return batch_kernel


class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        else:
            self.pad = nn.ReflectionPad2d(
                (kernel_size // 2, kernel_size // 2 - 1, kernel_size // 2, kernel_size // 2 - 1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B * C).view((B, C, H, W))


# DownSample
class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0),
                            torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1),
                            torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1 / 4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out


class BatchBicubic(nn.Module):
    def __init__(self, down_sample_scale):
        super(BatchBicubic, self).__init__()
        self.down_sample_scale = down_sample_scale
        self.bicubic = Bicubic()

    def __call__(self, hr):
        lr = self.bicubic(hr, scale=1 / self.down_sample_scale)
        return lr


# Noise
def gaussian_noise(noise_scale, width, height, std_min=0, std_max=4):
    std = random.uniform(std_min, std_max)
    gaussian_noise = torch.normal(mean=0, std=std, size=(1, width, height)).cuda()
    gaussian_noise = gaussian_noise * noise_scale
    return gaussian_noise


class BatchNoise(nn.Module):
    def __init__(self, noise_max):
        super(BatchNoise, self).__init__()
        self.noise_max = noise_max

    def __call__(self, lr_downsampled):
        B, C, H, W = lr_downsampled.size()  # B*N C H W
        noise_level = torch.rand(B, 1, 1, 1, 1).cuda() * self.noise_max
        noise = torch.randn_like(lr_downsampled).reshape(-1, 1, C, H, W).mul_(noise_level).view(-1, C, H, W)
        lr_noised = lr_downsampled.add_(noise)
        return lr_noised


class StableNoise(nn.Module):
    def __init__(self, noise):
        super(StableNoise, self).__init__()
        self.noise = noise

    def __call__(self, lr_downsampled):
        if self.noise == 0:
            return lr_downsampled
        else:
            _, C, H, W = lr_downsampled.size()
            noise_level = self.noise
            noise = torch.randn_like(lr_downsampled).reshape(1, 1, C, H, W).mul_(noise_level).view(1, C, H, W)
            lr_noised = lr_downsampled.add_(noise)
            return lr_noised


class BicubicDegradation(object):
    def __init__(self, scale):
        self.scale = scale
        self.down_sample = BatchBicubic(scale)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, W, H = hr_tensor.size()
            hr = hr_tensor.reshape(-1, C, W, H)
            # down_sample
            lr_downsampled = self.down_sample(hr)
            lr = torch.clamp(lr_downsampled.round(), 0, 255)
            if self.scale > 1:
                lr = lr.reshape(B, N, C, W // int(self.scale), H // int(self.scale))
            else:
                lr = lr.reshape(B, N, C, int(W // self.scale), int(H // self.scale))
            # print("lr shape", lr.shape)
            return lr


class IsoDegradation(object):
    def __init__(self, args):
        self.scale = args.scale
        self.kernel_size = args.kernel_size

        self.gen_kernel = BatchRandonKernel(args.blur_type_list, args.blur_type_prob, args.kernel_size,
                                            args.sigma_min, args.sigma_max)
        self.blur = BatchBlur(args.kernel_size)

        self.down_sample = BatchBicubic(self.scale)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            lr_downsampled = lr_downsampled.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_downsampled.round(), 0, 255)

            return lr


class StableIsoDegradation(object):
    def __init__(self, scale, sigma):
        self.scale = scale
        kernel_size = 21
        blur_type_list = ['iso_gaussian', 'aniso_gaussian']
        blur_type_prob = [1, 0]
        self.gen_kernel = BatchStableKernel(blur_type_list, blur_type_prob, kernel_size, sigma=sigma)
        self.blur = BatchBlur(kernel_size)

        self.down_sample = BatchBicubic(self.scale)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            lr_downsampled = lr_downsampled.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_downsampled.round(), 0, 255)

            return lr


class AnisoDegradation(object):
    def __init__(self, args):
        self.scale = args.scale
        self.kernel_size = args.kernel_size

        self.gen_kernel = BatchRandonKernel(args.blur_type_list, args.blur_type_prob, args.kernel_size, args.sigma_min,
                                            args.sigma_max)
        self.blur = BatchBlur(args.kernel_size)

        self.down_sample = BatchBicubic(args.scale)

        self.noise = BatchNoise(args.noise_max)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            # add noise
            # lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            lr_noised = self.noise(lr_downsampled)
            lr_noised = lr_noised.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_noised.round(), 0, 255)

            return lr


class StableAnisoDegradation(object):
    def __init__(self, theta, sigma_x, sigma_y, noise):
        self.scale = 4
        kernel_size = 21
        blur_type_list = ['iso_gaussian', 'aniso_gaussian']
        blur_type_prob = [0, 1]
        self.gen_kernel = BatchStableKernel(blur_type_list, blur_type_prob, kernel_size, 0.0,
                                            theta, sigma_x, sigma_y)
        self.blur = BatchBlur(kernel_size)

        self.down_sample = BatchBicubic(self.scale)

        self.noise = StableNoise(noise)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            # add noise
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            lr_noised = self.noise(lr_downsampled)
            lr_noised = lr_noised.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_noised.round(), 0, 255)

            return lr
