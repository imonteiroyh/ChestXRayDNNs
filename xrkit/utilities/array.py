import numpy as np


def handle_empty_array(array):
    if np.count_nonzero(array) == 0:
        array[0, 0] = 1

    return array
