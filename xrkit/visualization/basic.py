import cv2
import numpy as np
from IPython.display import Image, display

from xrkit.image_processing import resize_image


def display_image(image: np.ndarray, scale: float = 0.3) -> None:
    """
    Displays a resized image.

    Parameters:
        image:
            The image to be displayed.
        scale:
            The scaling factor.
    """

    resized_image = resize_image(image, scale)
    display(Image(data=cv2.imencode(".png", resized_image)[1].tobytes()))
