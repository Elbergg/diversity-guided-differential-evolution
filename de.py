from enum import Enum
from typing import Callable

import numpy as np


class Mode(Enum):
    EXPLORATION = 0
    EXPLOITATION = 1


def init_pop(
    psize: int,
    pdim: int,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
):
    pop = rng.uniform(low, high, size=(psize, pdim))
    return pop


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
    pop: np.ndarray[float], sample_size: int, i: int, rng: np.random.Generator
) -> np.ndarray[float]:
    indices = np.arange(len(pop))
    available_indices = np.delete(indices, i)
    chosen_indices = rng.choice(
        available_indices, size=sample_size, replace=False
    )
    return pop[chosen_indices]


def grade(el: np.ndarray[float], target_func: Callable) -> np.ndarray[float]:
    return target_func(el)


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
    F: float,
    debug: bool = False,
    seed: int = 110,
) -> np.ndarray[float]:
    rng = np.random.default_rng(seed)

    og_pop = init_pop(
        psize, pdim, search_space_lower_bounds, search_space_upper_bounds, rng
    )
    t = 0
    mode = Mode.EXPLOITATION
    while t < max_iter:
        work_pop = og_pop.copy()

        for i in range(psize):
            current_diversity = diversity(
                work_pop,
                upper_bounds=search_space_upper_bounds,
                lower_bounds=search_space_lower_bounds,
            )
            if current_diversity < dlow:
                if mode != Mode.EXPLORATION and debug:
                    print(
                        f"Switching to EXPLORATION at t={t} (Div: {current_diversity:.8f})"
                    )
                mode = Mode.EXPLORATION
            elif current_diversity > dhigh:
                if mode != Mode.EXPLOITATION and debug:
                    print(
                        f"Switching to EXPLOITATION at t={t} (Div: {current_diversity:.8f})"
                    )
                mode = Mode.EXPLOITATION

            p_i = work_pop[i]

            if mode == Mode.EXPLOITATION:
                c1, c2 = sample(work_pop, 2, i, rng)
                z = np.zeros_like(c1)
                for j in range(pdim):
                    if rng.uniform(0, 1) < cross_prob:
                        z[j] = c1[j]
                    else:
                        z[j] = c2[j]
                if grade(z, target_func) < grade(p_i, target_func):
                    work_pop[i] = z
            else:
                x1, x2, x3 = sample(work_pop, 3, i, rng)
                mutant = x3 + F * (x1 - x2)
                mutant = np.clip(
                    mutant,
                    search_space_lower_bounds,
                    search_space_upper_bounds,
                )
                if grade(mutant, target_func) < grade(p_i, target_func):
                    work_pop[i] = mutant

        og_pop = work_pop
        t += 1
    grades = [grade(x, target_func) for x in og_pop]
    return og_pop[np.argmin(grades)]
