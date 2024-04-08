import numpy as np
from skimage.metrics import structural_similarity as ssim


def structural_similarity_index_measure(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index Measure (SSIM) between two images.

    Parameters:
        image1 (np.ndarray):
            The first image.
        image2 (np.ndarray):
            The second image.

    Returns:
        float:
            The SSIM value.
    """

    return ssim(image1, image2, multichannel=True)
