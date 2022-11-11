import numpy as np


def psnr(im0, im1):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return -10 * np.log10(np.mean(np.power(im0 - im1, 2)))


def normalized_psnr(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return psnr(im0 / norm, im1 / norm)
