import random
from enum import Enum

import numpy as np
from ev_alg import init_pop

class Mode(Enum):
    EXPLORATION = 0,
    EXPLOITATION = 1

def diversity(pop: np.ndarray[int]) -> float:
    pass

def sample(pop: np.ndarray[int], i: int) -> np.ndarray[int]:
    return np.random.choice(pop, size=1, p=[1 if j != i else 0 for j in pop])


def grade() -> float:
    pass

def F(x1: np.ndarray[int],x2: np.ndarray[int])->np.ndarray[int]:
    return np.sum(x1 - x2)

def de_dg(psize: int, pdim: int, dlow: float, dhigh: float, cross_prob: float) -> np.ndarray[int]:
    og_pop = init_pop(psize, pdim)
    t = 0
    mode = Mode.EXPLORATION
    while(True):
        work_pop = og_pop.copy()
        for i in range(psize):
            if diversity(work_pop) < dlow:
                mode = Mode.EXPLORATION
            elif diversity(work_pop) > dhigh:
                mode = Mode.EXPLOITATION
            if mode == Mode.EXPLOITATION:
                p1 = work_pop[i]
                pw = sample(work_pop, i)
                c = p1
                for j in range(pdim):
                    if random.random() < cross_prob:
                        c[j] = pw[j]
                if grade(c) > grade(p1):
                    work_pop[i] = c
            else:
                x1 = sample(work_pop, i)
                x2 = sample(work_pop, i)
                mutant = work_pop[i] + F(x1, x2)
                work_pop[i] = mutant
        og_pop = work_pop
    return og_pop





