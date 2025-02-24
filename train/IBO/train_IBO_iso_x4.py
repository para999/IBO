from option.IBO.option_IBO_iso_x4 import args
from data.srdataset import SRDataset
from torch.utils.data import DataLoader
from utils import degradation, utility
from model.IBO.IBO import IBO
import numpy as np
import torch
import random
import os
import time


def main():
    if args.seed is not None:
        random.seed(args.seed)  # 随机生成一个种子
    torch.manual_seed(args.seed)  # 设置随机数种子

    # creat training & testing dataset and load
    dataset_train = SRDataset(args, name=args.data_train, train=True)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.n_threads, pin_memory=True, drop_last=True)

    dataset_test = SRDataset(args, name=args.data_test, train=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=args.n_threads,
                                 pin_memory=True, drop_last=False)

    # creat model
    print("Creating model...")
    model = IBO(args.scale).cuda()
    # loss & optim
    criterion_sr = torch.nn.L1Loss().cuda()
    optimizer = utility.make_optimizer(args, model)

    # path settings
    model_name = str(args.model_name)
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = args.project_path
    model_path = os.path.join(current_dir, 'experiment', model_name, 'model')
    log_path = os.path.join(current_dir, 'experiment', model_name, 'logs')
    log_file_path = os.path.join(log_path, 'log.txt')

    # resume
    if args.resume:
        if os.path.isfile(args.resume_path):
            checkpoint = torch.load(args.resume_path)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Resume from checkpoint {} , next epoch: {}".format(args.resume_path, args.start_epoch))
            with open(log_file_path, "a") as file:
                file.write("Resume from checkpoint {}, next epoch: {}\n".format(args.resume_path, args.start_epoch))
        else:
            args.start_epoch = 1
            print("checkpoint is not exist. Start training from epoch: {}".format(args.start_epoch))
    else:
        args.start_epoch = 1
        print("checkpoint is not exist. Start training from epoch: {}".format(args.start_epoch))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            with open(log_file_path, "w") as file:
                file.write("Start Training\n")
                file.write("Training Set is {}\n".format(args.data_train))
                file.write("Testing Set is {}\n".format(args.data_test))
                file.write("checkpoint is not exist. Start training from epoch: {}\n".format(args.start_epoch))

    # train eval and test
    for epoch in range(args.start_epoch, args.epochs_ibo + 1):
        # leaning rate settings
        adjust_learning_rate_sr(optimizer, epoch, args)
        # train and eval
        train(args, dataloader_train, model, criterion_sr, optimizer, epoch, log_file_path)

        # save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(current_dir, 'experiment', model_name, 'model', 'model_{:04d}.pth.tar'.format(epoch))
        torch.save(checkpoint, save_path)

        # test
        print("Epoch:{} Do test".format(epoch))
        test(args, dataloader_test, model, log_file_path, epoch)


def train(args, train_loader, model, criterion_blindsr, optimizer, epoch, log_file_path):
    # print training information and save
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    psnrs = AverageMeter('PSNR', ':.4e')

    progress_sr = ProgressMeter(
        len(train_loader) - 1,
        [batch_time, data_time, losses, psnrs],
        log_file_path,
        prefix="Epoch: [{}]".format(epoch))
    model.train()  # switch to the train mode

    end = time.time()
    # generate degraded image online
    degrade = degradation.IsoDegradation(args)
    for batch, (hr, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        hr = hr.cuda(non_blocking=True)

        # lr image
        hr = hr.unsqueeze(1)
        lr = degrade(hr)
        hr = hr[:, 0, ...]
        lr = lr[:, 0, ...]
        # print(lr.shape)
        # compute output
        if epoch <= args.epoch_uo:
            sr = model(lr, None, False)
        elif args.epoch_lo >= epoch > args.epoch_uo:
            sr = model(lr, hr, False)
        elif epoch > args.epoch_lo:
            sr = model(lr, None, True)

        loss = torch.tensor(0.0).cuda()
        loss_sr = criterion_blindsr(sr, hr)
        loss += loss_sr

        sr_eval = utility.quantize(sr, args.rgb_range)
        hr_eval = utility.quantize(hr, args.rgb_range)

        psnr = utility.calc_psnr(sr_eval, hr_eval, scale=args.scale, rgb_range=args.rgb_range)

        losses.update(loss.item())
        psnrs.update(psnr)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print
        if batch % args.print_every == 0:
            progress_sr.display(batch)


def test(args, test_loader, model, log_file_path, epoch):
    # test
    model.eval()

    with torch.no_grad():
        test_psnr = 0
        test_ssim = 0

        for batch, (hr, lr, _) in enumerate(test_loader):
            hr = hr.cuda()  # 1 c w h
            lr = lr.cuda()  # 1 c w/scale h/scale

            hr = crop_border_test(hr, args.scale)

            if epoch <= args.epoch_uo:
                sr = model(lr, None, False)
            elif args.epoch_lo >= epoch > args.epoch_uo:
                sr = model(lr, hr, False)
            elif epoch > args.epoch_lo:
                sr = model(lr, None, True)

            sr = utility.quantize(sr, args.rgb_range)
            hr = utility.quantize(hr, args.rgb_range)

            test_psnr += utility.calc_psnr(sr, hr, args.scale, args.rgb_range, benchmark=True)
            test_ssim += utility.calc_ssim(sr, hr, args.scale, benchmark=True)

        print("PSNR:{} SSIM:{}".format(test_psnr / len(test_loader), test_ssim / len(test_loader)))
        with open(log_file_path, "a") as file:
            file.write("PSNR:{} SSIM:{}\n".format(test_psnr / len(test_loader), test_ssim / len(test_loader)))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, log_file_path, prefix="", ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.log_file_path = log_file_path
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        with open(self.log_file_path, "a") as file:
            file.write('\t'.join(entries) + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate_sr(optimizer, epoch, args):
    lr = args.lr_sr * (args.gamma_sr ** ((epoch) // args.lr_decay_sr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def crop_border_test(img, scale):
    b, c, h, w = img.size()

    img = img[:, :, :int(h // scale * scale), :int(w // scale * scale)]

    return img


if __name__ == '__main__':
    main()
