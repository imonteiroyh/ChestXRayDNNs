import numpy as np


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.

    Parameters:
        image1 (np.ndarray):
            The first image.
        image2 (np.ndarray):
            The second image.

    Returns:
        float:
            The MSE value.
    """

    squared_difference = (image1 - image2) ** 2

    sum_of_squared_difference = np.sum(squared_difference)
    total_pixels = image1.shape[0] * image1.shape[1]
    error = sum_of_squared_difference / total_pixels

    return error
