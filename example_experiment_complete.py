#!/usr/bin/env python
"""A short yet complete example experiment script with restarts and batching.

Arguments
---------
This script must be called with 2-3 arguments:

    algorithm budget_multiplier [number_of_batches batch_to_execute]

``algorithm`` name of the algorithm to be benchamarke either 'scipy_de' or 'de_dg'

``budget_multiplier`` times dimension is the budget within which problem
instances are run repeatedly as long as too few successes are observed.

``batch_to_execute`` can only be omitted when ``number_of_batches == 1``.
When ``number_of_batches > 1``, the script must be executed repeatedly
(e.g., in parallel) with the different values for ``batch_to_execute =
0..number_of_batches-1`` to get data for the full experiment.

Usage
-----
To apply the code to a different solver/algorithm, `fmin` must be
re-assigned or re-defined accordingly, and the below code must be edited at
the two places marked with "### input" around lines 40 and 80.

See also: https://coco-platform.org/getting-started#experiment
"""

__author__ = "Nikolaus Hansen"
__copyright__ = "public domain"

import collections
import time

import mkl_bugfix  # noqa: F401 isort: skip

import cocoex
import numpy as np
import scipy

import de

### input: define suite and solver (see also "input" below where fmin is called)
suite_name = "bbob"  # filter for preliminary quick tests:
# suite_filter = "dimensions:2,3,5,10,20 instance_indices:1-5"    # "dimensions: 2,3,5,10,20 instance_indices:1-5"
suite_filter = ""

### reading in parameters
if __name__ == "__main__":
    import sys

    try:
        algorithm_choice = sys.argv[1].lower()
        if algorithm_choice not in ["scipy_de", "de_dg"]:
            raise ValueError("algorithm must be 'scipy_de' or 'de_dg'")
        budget_multiplier = float(sys.argv[2])
        number_of_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        batch_to_execute = int(sys.argv[4]) if len(sys.argv) > 4 else None

        if algorithm_choice == "scipy_de":
            fmin = scipy.optimize.differential_evolution
        else:
            fmin = de.de_dg
    except Exception as e:
        print(
            "Exception {} with calling arguments {}\n\n".format(e, sys.argv)
            + __doc__
        )
        raise

### prepare
suite = cocoex.Suite(
    suite_name, "", suite_filter
)  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = "{}_of_{}_{}D_on_{}{}".format(
    fmin.__name__,
    fmin.__module__ or "",
    int(budget_multiplier + 0.499),
    suite_name,
    ("_batch{:0" + str(len(str(number_of_batches - 1))) + "}of{}").format(
        batch_to_execute, number_of_batches
    )
    if number_of_batches > 1
    else "",
)
observer = cocoex.Observer(
    suite_name,
    # see https://numbbo.github.io/coco-doc/C/#observer-parameters
    "result_folder: {0}  algorithm_name: {1}".format(
        output_folder, fmin.__module__ + "." + fmin.__name__
    ),
)
repeater = cocoex.ExperimentRepeater(
    budget_multiplier,  # x dimension
    min_successes=0.75 * int(suite_filter.split("-")[1])
    if "indices:1-" in suite_filter
    else 11,
    max_sweeps=100,
)  # possible sweeps over the suite
batcher = cocoex.BatchScheduler(number_of_batches, batch_to_execute)
minimal_print = cocoex.utilities.MiniPrint()
timings = collections.defaultdict(list)  # key is the dimension
final_conditions = collections.defaultdict(
    list
)  # key is (id_fun, dimension, id_inst)
cocoex.utilities.write_setting(
    locals(), [observer.result_folder, "parameters.pydat"]
)

### go
time0 = time.time()
while not repeater.done():  # while budget is left and successes are few
    for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
        if not batcher.is_in_batch(problem) or repeater.done(problem):
            continue  # skip problem and bypass repeater.track
        problem.observe_with(observer)  # generate data for cocopp
        time1 = time.time()
        problem(problem.dimension * [0])  # for better comparability

        ### input: implement/amend the next few lines for another fmin
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
        else:
            raise ValueError("case for fmin={} not found".format(fmin))

        problem(xopt)  # make sure the returned solution is evaluated

        if (
            repeater._sweeps == 1
        ):  # time only the first (full) sweep through suite
            timings[problem.dimension].append(
                (time.time() - time1) / problem.evaluations
            )
        repeater.track(problem)  # track evaluations and final_target_hit
        minimal_print(problem)  # show progress
        final_conditions[problem.id_triple].append(
            repr([problem.evaluations, final_condition])
        )
        with open(
            observer.result_folder + "/final_conditions.pydict", "wt"
        ) as file_:
            file_.write(str(dict(final_conditions)).replace("],", "],\n"))

### final messaging
print(
    "\nTiming summary over all functions without repetitions:\n"
    "  dimension  median time [seconds/evaluation]\n"
    "  -------------------------------------"
)
for dimension in sorted(timings):
    ts = sorted(timings[dimension])
    print(
        "    {:3}       {:.1e}".format(
            dimension, (ts[len(ts) // 2] + ts[-1 - len(ts) // 2]) / 2
        )
    )
print("  -------------------------------------")

if number_of_batches > 1:
    print(
        "\n*** Batch {} of {} batches finished in {}."
        " Make sure to run *all* batches (0..{}) ***".format(
            batch_to_execute,
            number_of_batches,
            cocoex.utilities.ascetime(time.time() - time0),
            number_of_batches - 1,
        )
    )
else:
    print(
        "\n*** Full experiment done in %s ***"
        % cocoex.utilities.ascetime(time.time() - time0)
    )
print("    Data written into {}".format(observer.result_folder))

### post-process data
if number_of_batches == 1:
    print(
        "    Postprocess with 'python cocopp {} [...]'".format(
            observer.result_folder
        )
    )
    import cocopp  # post-processing module

    dsl = cocopp.main(
        observer.result_folder
    )  # re-run folders look like "...-001" etc
