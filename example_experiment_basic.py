import cocoex  # experimentation module
import cocopp

from de import de_dg

### input
suite_name = "bbob"
fmin = de_dg  # optimizer to be benchmarked
budget_multiplier = 5  # x dimension, increase to 3, 10, 30,...

### prepare
suite = cocoex.Suite(
    suite_name, "", "dimensions:2,3 instance_indices:1"
)  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = "{}_of_{}_{}D_on_{}".format(
    fmin.__name__, fmin.__module__ or "", int(budget_multiplier), suite_name
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
            psize=15 * problem.dimension,
            pdim=problem.dimension,
            dlow=0.0002,
            dhigh=0.25,
            cross_prob=0.5,
            search_space_lower_bounds=problem.lower_bounds,
            search_space_upper_bounds=problem.upper_bounds,
            max_iter=int(budget_multiplier * 20 * problem.dimension),
            F=0.5,
            debug=True,
        )
        problem(xopt)  # make sure the returned solution is evaluated
        repeater.track(problem)  # track evaluations and final_target_hit
        minimal_print(problem)  # show progress


        # print(f"Function: {problem.id_function}")
        # print(f"Best solution found: {xopt}")
        # print(f"Best value found: {problem.best_observed_fvalue1}")
        # print(f"Target value hit: {problem.final_target_hit}")
### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
