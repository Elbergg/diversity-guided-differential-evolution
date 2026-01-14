import de
import cocoex
import cocopp
from matplotlib import pyplot as plt
import scipy
import numpy as np

def create_convergence_chart(data, labels=['de_dg', 'scikit_de'], colors=['blue', 'orange']):
    for fun, label, color in zip(data, labels, colors):
        iters = list(range(1, len(fun[0].funvals) + 1))
        vals = np.mean(fun[0].funvals[:,1:],axis=1)
        std = np.std(fun[0].funvals[:,1:],axis=1)
        plt.plot(iters, vals, label=label, color=color)
        plt.fill_between(iters, vals-std, vals+std,
                         alpha=0.15, color=color,
                         label='±1 Standard Deviation')
        plt.plot(iters, vals-std, 'b--', alpha=0.5, linewidth=1)
        plt.plot(iters, vals+std, 'b--', alpha=0.5, linewidth=1)
    plt.title("Convergence on sphere function in dimension 5")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Values")
    plt.show()

def gather_data(fun):
    suite_name = "bbob"
    fmin = fun  # optimizer to be benchmarked
    budget_multiplier = 5  # x dimension, increase to 3, 10, 30,...
    dimension = 5
    budget_multiplier_2 = 40
    F = 0.5
    dlow = 0.05
    dhigh = 0.15
    cr = 0.5
    psize_multiplier = 15
    suite = cocoex.Suite(
        suite_name, "", f"dimensions:{dimension} function_indices:1 instance_indices:1-10"
    )

    output_folder = (
        f"exdata/{fmin.__name__}_{fmin.__module__}_IN_F={F}_dlow={dlow}_dhigh={dhigh}_"
        f"cr={cr}_psize={psize_multiplier}_bm2={budget_multiplier_2}of_{dimension}D_on_{suite_name}"
    )

    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
    minimal_print = cocoex.utilities.MiniPrint()

    while not repeater.done():  # while budget is left and successes are few
        for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
            if repeater.done(problem):
                continue  # skip this problem
            problem.observe_with(observer)  # generate data for cocopp
            problem(problem.dimension * [0])  # for better comparability
            if fmin == scipy.optimize.differential_evolution:
                res = fmin(
                    problem,
                    bounds=scipy.optimize.Bounds(
                        problem.lower_bounds, problem.upper_bounds
                    ),
                    strategy="rand1bin",
                    maxiter=int(budget_multiplier * 50 * problem.dimension),
                    popsize=15,
                    mutation=0.5,
                    recombination=0.5,
                    rng=np.random.default_rng(110),
                )
                xopt = res.x
                final_condition = (res.message, res.success)
            elif fmin == de.de_dg:
                res = fmin(
                    problem,
                    psize=15 * problem.dimension,
                    pdim=problem.dimension,
                    dlow=0.05,
                    dhigh=0.15,
                    cross_prob=0.5,
                    search_space_lower_bounds=problem.lower_bounds,
                    search_space_upper_bounds=problem.upper_bounds,
                    max_iter=int(budget_multiplier * 50 * problem.dimension),
                    F=0.5,
                    seed=110,
                )
                xopt = res
                final_condition = None
            problem(xopt)  # make sure the returned solution is evaluated
            repeater.track(problem)  # track evaluations and final_target_hit
            minimal_print(problem)  # show progress
    return observer.result_folder

### post-process data
# cocopp.main(observer.result_folder)
data_dg = cocopp.load2(gather_data(de.de_dg))
data_sci = cocopp.load2(gather_data(scipy.optimize.differential_evolution))
create_convergence_chart([data_dg, data_sci])

