import mkl_bugfix  # noqa: F401 isort: skip
import cocoex  # experimentation module
import cocopp

import de

### input
suite_name = "bbob"
fmin = de.de_dg  # optimizer to be benchmarked
budget_multiplier = 5  # x dimension, increase to 3, 10, 30,...

F = 0.5
dlow = 0.05
dhigh = 0.15
cr = 0.5
psize_multiplier = 15
budget_multiplier_2 = 40

### prepare
suite = cocoex.Suite(
    suite_name, "", "dimensions:2,3,5,10 instance_indices:1-5"
)  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = (
    f"{fmin.__name__}_{fmin.__module__}_IN_F={F}_dlow={dlow}_dhigh={dhigh}_"
    f"cr={cr}_psize={psize_multiplier}_bm2={budget_multiplier_2}of_{int(budget_multiplier)}D_on_{suite_name}"
)

observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
minimal_print = cocoex.utilities.MiniPrint()

### go
while not repeater.done():  # while budget is left and successes are few
    for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
        if repeater.done(problem):
            continue  # skip this problem
        problem.observe_with(observer)  # generate data for cocopp
        problem(problem.dimension * [0])  # for better comparability
        xopt = fmin(
            problem,
            psize=psize_multiplier * problem.dimension,
            pdim=problem.dimension,
            dlow=dlow,
            dhigh=dhigh,
            cross_prob=cr,
            search_space_lower_bounds=problem.lower_bounds,
            search_space_upper_bounds=problem.upper_bounds,
            max_iter=int(
                budget_multiplier * budget_multiplier_2 * problem.dimension
            ),
            F=F,
            debug=False,
            seed=110
        )
        problem(xopt)  # make sure the returned solution is evaluated
        repeater.track(problem)  # track evaluations and final_target_hit
        minimal_print(problem)  # show progress

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
