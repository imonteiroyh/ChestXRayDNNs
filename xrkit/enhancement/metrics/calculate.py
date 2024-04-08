from typing import Dict

import numpy as np

from xrkit.enhancement.metrics import (
    absolute_mean_brightness_error,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    root_mean_squared_error,
    structural_similarity_index_measure,
)


def calculate_enhancement_metrics(
    image1: np.ndarray, image2: np.ndarray, display_metrics: bool = False
) -> Dict[str, float]:
    """
    Calculate various metrics between two images and optionally display them.

    Parameters:
        image1 (np.ndarray):
            The first image.
        image2 (np.ndarray):
            The second image.
        display_metrics (bool):
            Whether to print the metrics. Defaults to False.

    Returns:
        Dict[str, float]:
            A dictionary containing the calculated metrics.
    """

    mse = mean_squared_error(image1, image2)
    rmse = root_mean_squared_error(mse)
    ambe = absolute_mean_brightness_error(image1, image2)
    ssim_value = structural_similarity_index_measure(image1, image2)
    psnr = peak_signal_to_noise_ratio(mse)

    metrics = {
        "Root Mean Squared Error (RMSE)": rmse,
        "Absolute Mean Brightness Error (AMBE)": ambe,
        "Structural Similarity Index Measure (SSIM)": ssim_value,
        "Peak Signal-to-Noise Ratio (PSNR)": psnr,
    }

    if display_metrics:
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    return metrics
