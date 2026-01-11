import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cocopp in batches")

    parser.add_argument("algorithms", type=str, choices=["scipy_de", "de_dg"])
    parser.add_argument("budget_multiplier", type=float)
    parser.add_argument("number_of_batches", type=int)

    args = parser.parse_args()
    processes = []

    for i in range(args.number_of_batches):
        proc = subprocess.Popen(
            [
                "python",
                "example_experiment_complete.py",
                args.algorithms,
                str(args.budget_multiplier),
                str(args.number_of_batches),
                str(i),
            ],
        )
        processes.append((i, proc))
        print(f"Started batch {i} (PID: {proc.pid})")

    results = []
    for batch_num, p in processes:
        return_code = p.wait()
        results.append((batch_num, return_code))
        print(f"[PID {p.pid}] End of batch {batch_num}", flush=True)

    for batch_num, exit_code in results:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (code: {exit_code})"
        print(f"Batch {batch_num}: {status}")
