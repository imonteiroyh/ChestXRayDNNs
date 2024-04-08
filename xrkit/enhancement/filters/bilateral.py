import cv2
import numpy as np

from xrkit.base import CONFIG
from xrkit.utilities import timer


@timer(return_execution_time=True)
def bilateral_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a bilateral filter to an image.

    This function applies a bilateral filter to the input image, which can reduce noise while preserving
    edges.

    Parameters:
        image (np.ndarray):
            The input image as a NumPy array.

    Returns:
        np.ndarray:
            The filtered image as a NumPy array.
    """

    bilateral_config = CONFIG.enhancement.bilateral
    filtered_image = cv2.bilateralFilter(
        image,
        d=bilateral_config.diameter,
        sigmaColor=bilateral_config.sigma_color,
        sigmaSpace=bilateral_config.sigma_space,
    )

    return filtered_image
