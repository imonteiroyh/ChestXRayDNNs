import numpy as np


def peak_signal_to_noise_ratio(mse: float) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) from the Mean Squared Error (MSE).

    Parameters:
        mse (float):
            The Mean Squared Error value.

    Returns:
        float:
            The PSNR value.
    """

    if mse == 0:
        return 0

    return -10 * np.log10(mse / (255) ** 2)
