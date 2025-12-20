from enum import Enum
from typing import Callable

import numpy as np

from ev_alg import init_pop


class Mode(Enum):
    EXPLORATION = 0
    EXPLOITATION = 1


def diversity(
    pop: np.ndarray[float],
    upper_bounds: np.ndarray[float],
    lower_bounds: np.ndarray[float],
) -> float:
    P_size = pop.shape[0]
    L = np.linalg.norm(upper_bounds - lower_bounds)
    avg = np.mean(pop, axis=0)
    scaling_factor = 1 / (L * P_size)
    distances = np.linalg.norm(pop - avg, axis=1)

    return scaling_factor * np.sum(distances)


def sample(
    pop: np.ndarray[float], sample_size: int, i: int
) -> np.ndarray[float]:
    indices = np.arange(len(pop))
    available_indices = np.delete(indices, i)
    chosen_indices = np.random.choice(
        available_indices, size=sample_size, replace=False
    )
    return pop[chosen_indices]


def grade(el: np.ndarray[float], target_func: Callable) -> np.ndarray[float]:
    return target_func(el)


# def F(x1: np.ndarray[float], x2: np.ndarray[float]) -> np.ndarray[float]:
#     return np.sum(x1, x2)


def de_dg(
    target_func: Callable,
    psize: int,
    pdim: int,
    dlow: float,
    dhigh: float,
    cross_prob: float,
    search_space_upper_bounds: np.ndarray[float],
    search_space_lower_bounds: np.ndarray[float],
    max_iter: int,
    F: int,
) -> np.ndarray[float]:
    og_pop = init_pop(psize, pdim)
    t = 0
    mode = Mode.EXPLORATION
    while t < max_iter:
        work_pop = og_pop.copy()
        # print(
        #     diversity(
        #         work_pop,
        #         upper_bounds=search_space_upper_bounds,
        #         lower_bounds=search_space_lower_bounds,
        #     )
        # )
        for i in range(psize):
            if (
                diversity(
                    work_pop,
                    upper_bounds=search_space_upper_bounds,
                    lower_bounds=search_space_lower_bounds,
                )
                < dlow
            ):
                mode = Mode.EXPLORATION
            elif (
                diversity(
                    work_pop,
                    upper_bounds=search_space_upper_bounds,
                    lower_bounds=search_space_lower_bounds,
                )
                > dhigh
            ):
                mode = Mode.EXPLOITATION
            # print(mode)
            if mode == Mode.EXPLOITATION:
                p1 = work_pop[i]
                pw = sample(work_pop, 1, i)[0]
                c = p1.copy()
                for j in range(pdim):
                    if np.random.uniform(0, 1) < cross_prob:
                        c[j] = pw[j]
                if grade(c, target_func) < grade(p1, target_func):
                    work_pop[i] = c
            else:
                x1, x2 = sample(work_pop, 2, i)
                mutant = work_pop[i] + F * (x1 - x2)
                work_pop[i] = mutant
        og_pop = work_pop
        t += 1
    # grades = grade(og_pop, target_func)
    grades = np.apply_along_axis(grade, 1, og_pop, target_func)
    return og_pop[np.argmin(grades)]
