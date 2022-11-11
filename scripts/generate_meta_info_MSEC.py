from os import path as osp
from PIL import Image
import imageio
import cv2
from basicsr.utils import scandir

# generated path meta_info
def generate_meta_info_ec_sub_GT():

    # gt_folder = 'datasets/MultiExposure_dataset/training/GT_IMAGES_x5_sub'
    # meta_info_txt = 'ec/data/meta_info/meta_info_ec_sub_GT.txt'

    lq_folder = 'datasets/MultiExposure_dataset/training/INPUT_IMAGES_sub'
    meta_info_txt = 'ec/data/meta_info/meta_info_ec_sub_lq_MSEC.txt'

    img_list = sorted(list(scandir(lq_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            # img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            img = cv2.imread(osp.join(lq_folder, img_path))  # lazy load
            width, height, n_channel = img.shape
            # mode = img.mode
            # if mode == 'RGB':
            #     n_channel = 3
            # elif mode == 'L':
            #     n_channel = 1
            # else:
            #     raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    generate_meta_info_ec_sub_GT()
