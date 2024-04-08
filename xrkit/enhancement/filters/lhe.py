import numpy as np
from skimage.filters import rank
from skimage.morphology import disk

from xrkit.base import CONFIG
from xrkit.utilities import timer


@timer(return_execution_time=True)
def local_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies local equalization to an image using a disk-shaped neighborhood.

    Parameters:
        image (np.ndarray):
            The input image to be equalized. It should be a grayscale image.

    Returns:
        np.ndarray:
            The equalized image, clipped to the range [0, 255] and converted to uint8 format.
    """
    local_equalization_config = CONFIG.enhancement.local_equalization

    neighborhood = disk(local_equalization_config.disk_footprint)
    equalized_image = rank.equalize(image, footprint=neighborhood)

    clipped_image = np.clip(equalized_image * 255, 0, 255).astype("uint8")

    return clipped_image
