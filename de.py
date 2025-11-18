import random
from enum import Enum
from typing import Callable
import numpy as np
from ev_alg import init_pop

class Mode(Enum):
    EXPLORATION = 0,
    EXPLOITATION = 1




def diversity(pop: np.ndarray[float], upper_bounds: np.ndarray[float], lower_bounds: np.ndarray[float]) -> float:
    L = np.linalg.norm(upper_bounds - lower_bounds)
    avg= np.mean(pop)

    return 1/(L*len(pop))*np.sqrt(np.sum(np.square(np.subtract(pop,avg)))) #is this correct? i have no idea

def sample(pop: np.ndarray[float], i: float) -> np.ndarray[float]:
    return np.random.choice(pop, size=1, p=[1 if j != i else 0 for j in pop])


def grade(el: np.ndarray[float], target_func: Callable) -> np.ndarray[float]:
    return target_func(el)

def F(x1: np.ndarray[float],x2: np.ndarray[float])->np.ndarray[float]:
    return np.sum(x1 - x2)

def de_dg(psize: int, pdim: int, dlow: float, dhigh: float, cross_prob: float, target_func: Callable, search_space_upper_bounds: np.ndarray[float], search_space_lower_bounds: np.ndarray[float], max_iter: int) -> np.ndarray[float]:
    og_pop = init_pop(psize, pdim)
    t = 0
    mode = Mode.EXPLORATION
    while t < max_iter:
        work_pop = og_pop.copy()
        for i in range(psize):
            if diversity(work_pop, upper_bounds=search_space_upper_bounds, lower_bounds=search_space_lower_bounds) < dlow:
                mode = Mode.EXPLORATION
            elif diversity(work_pop, upper_bounds=search_space_upper_bounds, lower_bounds=search_space_lower_bounds) > dhigh:
                mode = Mode.EXPLOITATION
            if mode == Mode.EXPLOITATION:
                p1 = work_pop[i]
                pw = sample(work_pop, i)
                c = p1
                for j in range(pdim):
                    if random.random() < cross_prob:
                        c[j] = pw[j]
                if grade(c, target_func)[0] > grade(p1, target_func)[0]:
                    work_pop[i] = c
            else:
                x1 = sample(work_pop, i)
                x2 = sample(work_pop, i)
                mutant = work_pop[i] + F(x1, x2)
                work_pop[i] = mutant
        og_pop = work_pop
    grades = grade(og_pop, target_func)
    return og_pop[np.argmax(grades)]





