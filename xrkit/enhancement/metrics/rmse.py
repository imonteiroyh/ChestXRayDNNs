import numpy as np


def root_mean_squared_error(mse: float) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) from the Mean Squared Error (MSE).

    Parameters:
        mse (float):
            The Mean Squared Error value.

    Returns:
        float:
            The RMSE value.
    """

    return np.sqrt(mse)
