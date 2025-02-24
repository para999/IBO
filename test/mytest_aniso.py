from option.IBO.option_IBO_test import args
from model.IBO.IBO import IBO
from utils import utility, degradation
from data.srdataset import SRDataset
from torch.utils.data import DataLoader
import torch
import random
import os


def load_model(model, model_path, model_name):
    if os.path.isfile(model_path):
        print("Loading model", model_name, "from", model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Test model", model_name)
    else:
        print("Model path does not exist:", model_path)


def main():
    if args.seed is not None:
        random.seed(args.seed)  # 随机生成一个种子
    torch.manual_seed(args.seed)  # 设置随机数种子

    model_paths = [
        'IBO/experiment/IBO_iso_x4/IBO_aniso_x4.pth.tar'

    ]
    model_names = [
        'IBO_aniso_x4'
    ]

    models = [IBO(args.scale).cuda() for _ in range(len(model_paths))]

    for model, model_path, model_name in zip(models, model_paths, model_names):
        load_model(model, model_path, model_name)

    # creat test dataset and load
    Test_List = ["Set5", 'Set14', "B100", "Urban100"]
    sigma_xs = [2.0, 2.0, 3.5, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0]
    sigma_ys = [0.2, 1.0, 1.5, 2.0, 2.0, 1.5, 2.0, 3.0, 4.0]
    thetas = [0, 10, 30, 45, 90, 120, 135, 165, 180]
    noises = [0, 5, 10]

    for name in Test_List:
        dataset_test = SRDataset(args, name=name, train=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                     drop_last=False)

        for i in range(0, len(sigma_xs)):
            sigma_x = sigma_xs[i]
            sigma_y = sigma_ys[i]
            theta = thetas[i]
            for j in range(0, len(noises)):
                noise = noises[j]
                degrade = degradation.StableAnisoDegradation(theta, sigma_x, sigma_y, noise)
                print(f"Degradation parameters:")
                print(f"Sigma X: {sigma_x}")
                print(f"Sigma Y: {sigma_y}")
                print(f"Theta: {theta}")
                print(f"Noise: {noise}")

                model_list = models[:]

                batch_deg_test(dataloader_test, model_list, args, degrade)


def batch_deg_test(test_loader, model_list, args, degrade):
    with torch.no_grad():
        test_psnr_list = [0] * len(model_list)
        test_ssim_list = [0] * len(model_list)

        for batch, (hr, _) in enumerate(test_loader):
            hr = hr.cuda(non_blocking=True)

            hr = crop_border_test(hr, args.scale)
            hr = hr.unsqueeze(1)

            lr = degrade(hr)
            hr = hr[:, 0, ...]
            lr = lr[:, 0, ...]

            hr = utility.quantize(hr, args.rgb_range)

            for i, model in enumerate(model_list):
                model.eval()

                sr = model(lr, hr=None, sr=True)
                sr = utility.quantize(sr, args.rgb_range)

                test_psnr_list[i] += utility.calc_psnr(sr, hr, args.scale, args.rgb_range, benchmark=True)
                test_ssim_list[i] += utility.calc_ssim(sr, hr, args.scale, benchmark=True)
        for i in range(len(model_list)):
            print("{:.2f}/{:.4f}".format(test_psnr_list[i] / len(test_loader),
                                         test_ssim_list[i] / len(test_loader)))


def crop_border_test(img, scale):
    b, c, h, w = img.size()

    img = img[:, :, :int(h // scale * scale), :int(w // scale * scale)]

    return img


if __name__ == '__main__':
    main()
