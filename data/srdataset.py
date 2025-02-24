import glob
import pickle
import imageio
import os
import torch
from torch.utils.data import Dataset
from data import common


class SRDataset(Dataset):

    def __init__(self, args, name, train=True):
        super(SRDataset, self).__init__()
        self.args = args
        self.name = str(name)
        self.train = train
        self.data_dir = args.data_dir

        if self.train:  # train mode
            # patch settings for training
            self.repeat = args.repeat
            self.patch_cut = args.patch_cut
            self.patch_cut_two = args.patch_cut_two
            # dir settings
            self.path = os.path.join(self.data_dir, self.name)  # dataset/DF2K
            self.hr_path = os.path.join(self.path, 'HR')  # dataset/DF2K/HR
            print("Load training dataset. path(HR):", self.hr_path)
            self.bin_path = os.path.join(self.path, 'bin')  # dataset/DF2K/bin
            os.makedirs(self.bin_path, exist_ok=True)
            self.bin_hr_path = self.hr_path.replace(self.path, self.bin_path)  # dataset/DF2K/bin/HR
            os.makedirs(self.bin_hr_path, exist_ok=True)
            pattern = os.path.join(self.hr_path, '*.[pj][np][gg]')
            self.list_hr = sorted(glob.glob(pattern))  # dataset/DF2K/HR/0001.png

            self.list_bin_hr = []
            for hr in self.list_hr:
                bin_hr = hr.replace(self.path, self.bin_path)
                bin_hr = bin_hr.replace('.png', '.pt').replace('.jpg', '.pt')
                self.list_bin_hr.append(bin_hr)
                if os.path.isfile(bin_hr):
                    pass
                else:
                    print('{} do not exist. Making...'.format(os.path.basename(bin_hr)))
                    image = [{
                        'name': os.path.splitext(os.path.basename(hr))[0],
                        'image': imageio.v3.imread(hr)
                    }]
                    with open(bin_hr, 'wb') as b:
                        pickle.dump(image, b)
        # test mode
        else:
            # image path
            self.path = os.path.join(self.data_dir, self.name)
            self.hr_path = os.path.join(self.path, 'HR')
            self.lr_path = os.path.join(self.path, 'LR')
            print("Load testing datatest. path(HR):", self.hr_path)
            print("Load testing datatest. path(LR):", self.lr_path)
            # bin image path
            self.bin_path = os.path.join(self.path, 'bin')
            os.makedirs(self.bin_path, exist_ok=True)
            self.bin_hr_path = self.hr_path.replace(self.path, self.bin_path)
            self.bin_lr_path = self.lr_path.replace(self.path, self.bin_path)
            os.makedirs(self.bin_hr_path, exist_ok=True)
            os.makedirs(self.bin_lr_path, exist_ok=True)
            pattern = os.path.join(self.hr_path, '*.[pj][np][gg]')
            self.list_hr = sorted(glob.glob(pattern))

            self.list_bin_hr = []
            for hr in self.list_hr:
                bin_hr = hr.replace(self.path, self.bin_path)
                bin_hr = bin_hr.replace('.png', '.pt').replace('.jpg', '.pt')
                self.list_bin_hr.append(bin_hr)
                if os.path.isfile(bin_hr):
                    pass
                else:
                    print('{} do not exist. Making...'.format(os.path.basename(bin_hr)))
                    image = [{
                        'name': os.path.splitext(os.path.basename(hr))[0],
                        'image': imageio.v3.imread(hr)
                    }]
                    with open(bin_hr, 'wb') as b:
                        pickle.dump(image, b)
            pattern = os.path.join(self.lr_path, '*.[pj][np][gg]')
            self.list_lr = sorted(glob.glob(pattern))

            self.list_bin_lr = []
            for lr in self.list_lr:
                bin_lr = lr.replace(self.path, self.bin_path)
                bin_lr = bin_lr.replace('.png', '.pt').replace('.jpg', '.pt')
                self.list_bin_lr.append(bin_lr)
                if os.path.isfile(bin_lr):
                    pass
                else:
                    print('{} do not exist. Making...'.format(os.path.basename(bin_lr)))
                    image = [{
                        'name': os.path.splitext(os.path.basename(lr))[0],
                        'image': imageio.v3.imread(lr)
                    }]
                    with open(bin_lr, 'wb') as b:
                        pickle.dump(image, b)

    def __len__(self):
        # get len of training data
        if self.train:
            return len(self.list_hr) * self.repeat
        # get len of testing data
        else:
            return len(self.list_hr)

    def __getitem__(self, idx):
        # get training data
        if self.train:
            idx = idx % len(self.list_bin_hr)
            bin_hr_dir = self.list_bin_hr[idx]
            filename = os.path.splitext(os.path.basename(bin_hr_dir))[0]
            with open(bin_hr_dir, 'rb') as bin_hr_dir:
                hr = pickle.load(bin_hr_dir)[0]['image']
            if self.patch_cut:
                if self.patch_cut_two:
                    hr = self.get_two_patch(hr)
                    hr = [common.set_channel(img, n_channels=self.args.n_colors) for img in hr]
                    # hr type = list, len = 2, shape = HWC if patch_cut == False else shape = [patch size, ~, C]
                    hr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range) for img in hr]
                    #  type = list  shape = [2, C, patch size, ~]
                    hr_tensor = torch.stack(hr_tensor, 0)  # type = tensor, shape = [2, C, patch size, ~]
                else:
                    hr = self.get_patch(hr)
                    hr = common.set_channel(hr, n_channels=self.args.n_colors)  # [patch size, ~, C]
                    hr_tensor = common.np2Tensor(hr, rgb_range=self.args.rgb_range)  # [C, patch size, ~]
            else:
                hr = common.set_channel(hr, n_channels=self.args.n_colors)
                hr_tensor = common.np2Tensor(hr, rgb_range=self.args.rgb_range)
            return hr_tensor, filename
        # get testing data
        else:
            bin_hr_dir = self.list_bin_hr[idx]
            bin_lr_dir = self.list_bin_lr[idx]

            filename = os.path.splitext(os.path.basename(bin_hr_dir))[0]
            with open(bin_hr_dir, 'rb') as hr:
                hr = pickle.load(hr)[0]['image']
            with open(bin_lr_dir, 'rb') as lr:
                lr = pickle.load(lr)[0]['image']
            hr = [common.set_channel(img, n_channels=self.args.n_colors) for img in [hr]]
            lr = [common.set_channel(img, n_channels=self.args.n_colors) for img in [lr]]

            hr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range) for img in hr]
            lr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range) for img in lr]

            return hr_tensor[0], lr_tensor[0], filename

    def get_patch(self, hr):
        hr_patch = common.get_patch(hr, patch_size=self.args.patch_size, scale=self.args.scale)  # h w c
        if self.args.augment:  # default false
            hr_patch = common.augment(hr_patch, hflip=True, rot=True)  # include flip and rotation
        return hr_patch

    def get_two_patch(self, hr):
        out = []
        for _ in range(2):
            hr_patch = common.get_patch(hr, patch_size=self.args.patch_size, scale=self.args.scale)
            if self.args.augment:
                hr_patch = common.augment(hr_patch, hflip=True, rot=True)  # include flip and rotation
            out.append(hr_patch)
        return out
