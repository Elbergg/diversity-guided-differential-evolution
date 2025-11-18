import numpy as np

def init_pop(psize: int, pdim:int):
    pop = np.random.randint(
        0, 2, size=(psize, pdim)
    )
    return pop
