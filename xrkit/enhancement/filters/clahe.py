import cv2
import numpy as np

from xrkit.base import CONFIG
from xrkit.utilities import timer


@timer(return_execution_time=True)
def contrast_limited_adaptative_histogram_equalization(input_image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    This function applies CLAHE to the input image, which can improve the contrast of the image by limiting
    the contrast enhancement to the local neighborhood of each pixel.

    Parameters:
        input_image (np.ndarray):
            The input image as a NumPy array.

    Returns:
        np.ndarray:
            The image with CLAHE applied as a NumPy array.
    """

    clahe_config = CONFIG.enhancement.clahe
    clahe_transformer = cv2.createCLAHE(
        clipLimit=clahe_config.clip_limit, tileGridSize=clahe_config.tile_grid_size
    )
    equalized_image = clahe_transformer.apply(input_image)

    return equalized_image
