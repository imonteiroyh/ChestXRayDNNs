import cv2
import numpy as np

from xrkit.utilities import timer


@timer(return_execution_time=True)
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to an image.

    This function applies histogram equalization to the input image, which can improve the contrast of the
    image.

    Parameters:
        image (np.ndarray):
            The input image as a NumPy array.

    Returns:
        np.ndarray:
            The equalized image as a NumPy array.
    """

    equalized_image = cv2.equalizeHist(image)

    return equalized_image
