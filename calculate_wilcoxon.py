import csv

import cocopp
import numpy as np
from scipy.stats import wilcoxon

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
TARGET_PRECISION = 1e-8
OUTPUT_CSV_FILE = "wilcoxon_results.csv"

ds_scipy = cocopp.load2("exdata/de_scipy_5D_full_seeded")
ds_de_dg = cocopp.load2(
    "exdata/de_dg_of_de_5D_on_bbob_batch_full_all_dim_seeded"
)


def get_data_map(ds_list):
    data_map = {}
    for ds in ds_list:
        key = (ds.dim, ds.funcId)
        if key not in data_map:
            data_map[key] = []

        data_map[key].extend(ds.readfinalFminusFtarget)
    return data_map


map_scipy = get_data_map(ds_scipy)
map_de_dg = get_data_map(ds_de_dg)

func_ids = sorted(list(set(f for _, f in map_scipy.keys())))
dimensions = sorted(list(set(d for d, _ in map_scipy.keys())))
csv_data = []

print("\nAGGREGATED OVER FUNCTIONS (per Dimension)")
print("W-T-L: Scipy Wins (Lower Error) - Ties - DE_DG Wins (Lower Error)")
print("-" * 95)
print(
    f"{'Dim':<5} | {'#Funcs':<7} | {'Scipy Median':<15} | {'DE_DG Median':<15} | {'W-T-L':<10} | {'p-value'}"
)
print("-" * 95)

for dim in dimensions:
    scipy_func_vals = []
    dedg_func_vals = []

    wins = 0
    ties = 0
    losses = 0

    for f in func_ids:
        key = (dim, f)

        if key not in map_scipy or key not in map_de_dg:
            continue

        val_scipy = max(np.median(map_scipy[key]), TARGET_PRECISION)
        val_dedg = max(np.median(map_de_dg[key]), TARGET_PRECISION)

        scipy_func_vals.append(val_scipy)
        dedg_func_vals.append(val_dedg)

        if val_scipy < val_dedg:
            wins += 1
        elif val_scipy > val_dedg:
            losses += 1
        else:
            ties += 1

    _, p_val = wilcoxon(
        scipy_func_vals,
        dedg_func_vals,
        alternative="two-sided",
        zero_method="wilcox",
    )

    p_str = f"{p_val:.4f}"
    if p_val < 0.05:
        p_str = f"{GREEN}{p_str}{RESET}"
    else:
        p_str = f"{RED}{p_str}{RESET}"

    print(
        f"{dim:<5} | {len(scipy_func_vals):<7} | "
        f"{np.median(scipy_func_vals):<15.2e} | "
        f"{np.median(dedg_func_vals):<15.2e} | "
        f"{f'{wins}-{ties}-{losses}':<10} | {p_str}"
    )

    row = {
        "Dimension": dim,
        "Num_Functions": len(scipy_func_vals),
        "Scipy_Median": np.median(scipy_func_vals),
        "DE_DG_Median": np.median(dedg_func_vals),
        "Wins": wins,
        "Ties": ties,
        "Losses": losses,
        "P_Value": p_val,
        "Significant": "Yes" if (p_val < 0.05) else "No",
    }
    csv_data.append(row)


headers = list(csv_data[0].keys())
with open(OUTPUT_CSV_FILE, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(csv_data)
print(f"\n{GREEN}Results successfully saved to {OUTPUT_CSV_FILE}{RESET}")
