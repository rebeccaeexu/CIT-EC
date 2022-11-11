import os
import os.path as osp
import glob
import cv2
import numpy as np

from basicsr.utils import scandir


def multiframe_paired_paths_from_folders(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    lq_folder, gt_folder = folders

    lq_paths = list(scandir(lq_folder))

    lq_names = []
    for lq_path in lq_paths:
        lq_name = osp.basename(lq_path).split('.JPG')[0]
        lq_names.append(lq_name)

    paths = []

    for lq_name in lq_names:
        lq_path = osp.join(lq_folder, lq_name + '.JPG')
        key = osp.basename(lq_name).split('_')
        gt_name = lq_name.rstrip(key[-1]).rstrip('_')  # a0001-jmac_DSC1459
        gt_path = osp.join(gt_folder, gt_name + '.jpg')    ## MSEC
        # gt_path = osp.join(gt_folder, gt_name + '.JPG')    ## SICE
        paths.append(dict([('lq_path', lq_path), ('gt_path', gt_path), ('key', lq_name)]))
    return paths


def multiframe_paired_paths_from_meta_info_file(folders, meta_info_file):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    input_folder, gt_folder = folders

    with open(meta_info_file, 'r') as fin:
        lq_names = [line.strip().split('.JPG')[0] for line in fin]

    paths = []

    for lq_name in lq_names:

        lq_key = osp.basename(lq_name[:-5]).split('_')
        sub = (lq_name[-5:])
        gt_name = lq_name[:-5].rstrip(lq_key[-1]).rstrip('_')  # a0001-jmac_DSC1459
        lq_path = osp.join(input_folder, lq_name + '.JPG')
        gt_path = osp.join(input_folder.replace('INPUT', 'GT'), gt_name + sub + '.jpg')  ## for MSEC dataset
        # gt_path = osp.join(input_folder.replace('lq', 'gt'), gt_name + sub + '.JPG')  ## for SICE dataset
        paths.append(dict([('lq_path', lq_path), ('gt_path', gt_path), ('key', lq_name)]))
    return paths



def multiframe_paths_from_folders(folders):
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. 'f'But got {len(folders)}')
    lq_folder, gt_folder = folders

    lq_paths = list(scandir(lq_folder))

    lq_names = []
    lq_suffixs = []
    for lq_path in lq_paths:
        lq_name = osp.basename(lq_path).split('.')[0]
        suffix = osp.basename(lq_path).split('.')[1]
        lq_names.append(lq_name)
        lq_suffix = osp.join(lq_name + '.' + suffix)
        lq_suffixs.append(lq_suffix)

    paths = []

    for lq_suffix in lq_suffixs:
        lq_name = osp.basename(lq_suffix).split('.')[0]
        lq_path = osp.join(lq_folder, lq_suffix)
        gt_path = lq_path
        paths.append(dict([('lq_path', lq_path), ('gt_path', gt_path), ('key', lq_name)]))
    return paths


def tensor2numpy(tensor):
    img_np = tensor.squeeze().numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)

