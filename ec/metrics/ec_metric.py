import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY
from skimage.measure import compare_ssim


@METRIC_REGISTRY.register()
def calculate_ec_psnr(img, img2):
    normalized_psnr = -10 * np.log10(np.mean(np.power(img - img2, 2)))
    if normalized_psnr == 0:
        return float('inf')
    return normalized_psnr


@METRIC_REGISTRY.register()
def calculate_ec_ssim(img, img2):
    ec_ssim = compare_ssim(img, img2, multichannel=True)
    return ec_ssim
