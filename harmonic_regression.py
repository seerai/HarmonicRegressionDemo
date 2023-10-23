"""Harmonic Regression in Landsat Imagery

Based on the techniques used in: https://www.sciencedirect.com/science/article/abs/pii/S0924271618300066?via%3Dihub

"""
import logging
from tesseract import serve, get_model_args
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG)


class Model:
    def __init__(self):
        args = get_model_args()
        self.forder = args.get("forder", 4)
        self.period = args.get("period", 365.2421891)
        self.n_params = 2 * self.forder + 2

    def inference(self, assets: dict, grids: dict, logger: logging.Logger, **kwargs) -> dict:
        # check that the number of time steps is greater than the number of parameters
        if assets["$0"].shape[0] < self.n_params:
            logger.error(
                f"Fit is underconstrained. Number of time steps ({assets['landsat'].shape[0]}) must be greater than the number of parameters ({self.n_params})"
            )
            raise ValueError(
                f"Fit is underconstrained. Number of time steps ({assets['landsat'].shape[0]}) must be greater than the number of parameters ({self.n_params})"
            )
        logger.info("Running tasseled cap transformation...")
        tc_data = tasseled_cap(assets["landsat"])

        times = grids["$0"]["t"][:, 0].astype("datetime64[D]").astype(int)

        logger.info("Running regression...")
        x = lstsq(times, tc_data, order=self.forder, censored=True)
        x = np.expand_dims(x, axis=0)
        return {
            "brightness_params": x[:, :, 0, :, :],
            "greenness_params": x[:, :, 1, :, :],
            "wetness_params": x[:, :, 2, :, :],
        }

    def get_model_info(self):
        return {
            "inputs": [{"name": "landsat", "dtype": "i2", "shape": [200, 6, 256, 256]}],
            "outputs": [
                {
                    "name": "brightness_params",
                    "dtype": "<f8",
                    "shape": [1, self.n_params, 256, 256],
                },
                {
                    "name": "greenness_params",
                    "dtype": "<f8",
                    "shape": [1, self.n_params, 256, 256],
                },
                {
                    "name": "wetness_params",
                    "dtype": "<f8",
                    "shape": [1, self.n_params, 256, 256],
                },
            ],
        }


def fourier_matrix(times, order=4, P=365.2421891):
    """Create a matrix of fourier series for a given order and times.

    This matrix is used in the least squares fit. It is the matrix A in the equation Ax = b.
    There should be 2*order+1 columns in the matrix. Each row will be a fourier series
    for a given time.

    Args:
        times (np.ndarray): 1D array of times
        order (int, optional): Order of the fourier series. Defaults to 4.
    """
    nt = len(times)  # number of time steps
    A = np.ones((nt, (2 * order + 2)))  # A[0] = 1 is the constant term
    A[:, 1] = times  # A[1] = t is the linear term
    n = 1
    for i in range(2, (2 * order) + 2, 2):
        A[:, i] = np.sin(2 * n * np.pi * times / P)
        A[:, i + 1] = np.cos(2 * n * np.pi * times / P)
        n += 1
    return A


def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Code taken from blog: https://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    If you are masking out np.nan values, you will need to set them to another number. These
    indices will be ignored because they are multiplied by 0, but if they are left as NaNs,
    the whole fit on that row will be NaN.

    Args:
        A (ndarray) : m x r matrix
        B (ndarray) : m x n matrix
        M (ndarray) : m x n binary masking matrix (zeros indicate missing values)

    Returns:
        X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.leastsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
    T = np.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # n x r x r tensor
    return np.squeeze(np.linalg.solve(T, rhs)).T  # transpose to get r x n


def lstsq(times, data, order=4, P=365.2421891, censored=True):
    """Least squares fit of a fourier series to a target using either scipy.linalg.lstsq or the
    censored least squares method.

    Args:
        times (np.ndarray): 1D array of times
        data (np.ndarray): 4D array of data to fit. Axes are (time, band, y, x)
        order (int, optional): Order of the fourier series. Defaults to 4.
        P (float, optional): Period of the fourier series in whatever units your data
           appears in. Defaults to 365.2421891 which is one siderial year in days.
        censored (bool, optional): Whether to use censored least squares. Use this
           option if there are NaN values (or 0 in the case of Int) in the data array. Defaults to True.
    """
    A = fourier_matrix(times, order=order, P=P)
    t, b, y, x = data.shape
    data = np.moveaxis(data, 0, -1)
    data = data.reshape(y * x * b, t)
    if censored:
        # M = ~np.isnan(data)
        M = data.astype("bool").astype(float)  # fill_value=0 so mask on 0 in the data array
        # np.nan_to_num(data, copy=False, nan=-1.0)
        params = censored_lstsq(A, data.T, M.T)
    else:
        params, _, _, _ = np.linalg.lstsq(A, data.T, rcond=None)
    return params.reshape(2 * order + 2, b, y, x)


def tasseled_cap(data: np.ndarray) -> np.ndarray:
    """Tasseled Cap Transformation (TCT) is a linear transformation of Landsat 8 bands.

    Coefficients obtained from the paper: https://www.tandfonline.com/doi/full/10.1080/2150704X.2014.915434
    Can also be found at: https://yceo.yale.edu/tasseled-cap-transform-landsat-8-oli
    """
    tc_xform = np.array(
        [  # blue, green, red, NIR, SWIR1, SWIR2
            [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],  # Brightness
            [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],  # Greenness
            [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],  # Wetness
        ],
        dtype=np.float16,
    )
    return np.einsum("ij,ljno->lino", tc_xform, data)


if __name__ == "__main__":
    model = Model()
    serve(model.inference, model.get_model_info)
