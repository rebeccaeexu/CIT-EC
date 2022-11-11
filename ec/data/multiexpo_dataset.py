import torch
from torch.utils import data as data
import numpy as np

from ec.data.data_util import *
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class MultiExpoImageDataset(data.Dataset):

    def __init__(self, opt):
        super(MultiExpoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = multiframe_paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], self.opt['meta_info_file'])
        else:
            self.paths = multiframe_paths_from_folders([self.lq_folder, self.gt_folder])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.paths[index]['key']

        # Load gt image and alignratio.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # Load lq images and exposures.
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq_s')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # img_lqs: (c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lq, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.paths)
