"""Harmonic Regression in Landsat Imagery

Based on the techniques used in: https://www.sciencedirect.com/science/article/abs/pii/S0924271618300066?via%3Dihub

"""
import logging
from geodesic.tesseract.models import serve
import numpy as np
from scipy.optimize import fmin

logging.basicConfig(level=logging.DEBUG)


class Model:
    def __init__(self):
        self.forder = 3
        self.TCT_bands = 3
        self.n_params = (2 * self.forder + 1) * self.TCT_bands
        pass

    def inference(self, assets: dict, logger: logging.Logger) -> dict:
        logger.info("Running tasselled cap transformation...")
        # Tasselled cap
        tc_data = self.tasseled_cap(assets)
        t, b, y, x = tc_data.shape

        logger.info("Running regression...")
        # Regression
        start_params = np.ones((self.n_params))
        output = np.empty((1, self.n_params, y, x))
        for i, row in enumerate(tc_data):
            for j, col in enumerate(row):
                min_params = fmin(fit_func, start_params, args=(times, target, self.forder))

        return {
            'fit_params': np.array(),

        }

    def get_model_info(self):
        return {
            'inputs': [
                {
                    'name': 'landsat',
                    'dtype': 'i2',
                    'shape': [48, 7, 256, 256]
                }
            ],
            'outputs': [
                {
                    'name': 'fit_params',
                    'dtype': 'f2',
                    'shape': [1, 3, 256, 256]
                }
            ]
        }


def fourier_series(x: np.ndarray, t: np.ndarray, order=1) -> np.float64:
    n = 365.2421891  # siderial days per year. Same as used in https://www.sciencedirect.com/science/article/abs/pii/S0924271618300066?via%3Dihub
    y = x[0]
    order = 1
    for i in range(1, (2*order)+1, 2):
        y = y + x[i]*np.sin(order*np.pi*t/n) + x[i+1]*np.cos(order*np.pi*t/n)
        order += 1
    return y


def fit_func(x, t, target, order):
    pred = fourier_series(x, t, order)
    return np.sum((pred - target)**2)


def tasseled_cap(data: np.ndarray) -> np.ndarray:
    """Tasseled Cap Transformation (TCT) is a linear transformation of Landsat 8 bands.

    Coefficients obtained from the paper: https://www.tandfonline.com/doi/full/10.1080/2150704X.2014.915434
    Can also be found at: https://yceo.yale.edu/tasseled-cap-transform-landsat-8-oli
    """
    tc_xform = np.array(
        [  # blue, green, red, NIR, SWIR1, SWIR2
            [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],  # Brightness
            [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],  # Greenness
            [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]  # Wetness
        ]
    )
    return np.einsum('ij,ljno->lino', tc_xform, data)


if __name__ == '__main__':
    model = Model()
    serve(model.inference, model.get_model_info)
