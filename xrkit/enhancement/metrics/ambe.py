import cv2
import numpy as np
from scipy.ndimage import mean


def absolute_mean_brightness_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Absolute Mean Brightness Error (AMBE) between two images.

    Parameters:
        image1 (np.ndarray):
            The first image.
        image2 (np.ndarray):
            The second image.

    Returns:
        float:
            The AMBE value.
    """

    img1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
    img2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2

    mean_brightness1 = mean(img1)
    mean_brightness2 = mean(img2)

    return np.abs(mean_brightness1 - mean_brightness2)
