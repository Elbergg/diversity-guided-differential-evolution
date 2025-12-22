import numpy as np


def init_pop(psize: int, pdim: int, low: np.ndarray, high: np.ndarray):
    pop = np.random.uniform(low, high, size=(psize, pdim))
    return pop
