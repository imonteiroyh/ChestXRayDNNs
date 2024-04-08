import numpy as np
from skimage.restoration import denoise_tv_chambolle

from xrkit.base import CONFIG
from xrkit.utilities import timer


@timer(return_execution_time=True)
def total_variance_denoising(image: np.ndarray) -> np.ndarray:
    total_variance_config = CONFIG.enhancement.total_variance
    denoised_image = denoise_tv_chambolle(
        image,
        weight=total_variance_config.weight,
    )

    clipped_image = np.clip(denoised_image * 255, 0, 255).astype("uint8")

    return clipped_image
