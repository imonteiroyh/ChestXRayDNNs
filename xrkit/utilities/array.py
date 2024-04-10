import numpy as np


def handle_empty_array(array: np.ndarray) -> np.ndarray:
    """
    Handle empty arrays by filling them with a default value if they contain only zeros.

    Args:
        array (np.ndarray):
            Input array.

    Returns:
        np.ndarray:
            Array with zeros replaced by a default value if the array was empty.
    """

    if np.count_nonzero(array) == 0:
        array[0, 0] = 1

    return array
