import argparse
import os
from typing import Dict

import numpy as np


def summarize_xp_matrix(xp_matrix: np.ndarray) -> Dict:
    if xp_matrix.ndim != 2 or xp_matrix.shape[0] != xp_matrix.shape[1]:
        raise ValueError(
            f"Expected a square XP matrix, got shape {xp_matrix.shape}."
        )

    num_agents = xp_matrix.shape[0]
    diagonal_mask = np.eye(num_agents, dtype=bool)
    off_diagonal_mask = ~diagonal_mask

    sp_returns = np.asarray(np.diag(xp_matrix), dtype=np.float64)
    xp_returns = np.asarray(xp_matrix[off_diagonal_mask], dtype=np.float64)
    sp_returns = sp_returns[np.isfinite(sp_returns)]
    xp_returns = xp_returns[np.isfinite(xp_returns)]

    def mean_and_se(values: np.ndarray):
        if values.size == 0:
            return np.nan, np.nan
        mean = float(values.mean())
        if values.size <= 1:
            return mean, np.nan
        se = float(values.std(ddof=1) / np.sqrt(values.size))
        return mean, se

    average_sp, standard_error_sp = mean_and_se(sp_returns)
    average_xp, standard_error_xp = mean_and_se(xp_returns)

    return {
        "average_sp": average_sp,
        "standard_error_sp": standard_error_sp,
        "average_xp_excluding_self": average_xp,
        "standard_error_xp_excluding_self": standard_error_xp,
        "num_agents": int(num_agents),
        "num_sp_pairs": int(sp_returns.size),
        "num_xp_pairs_excluding_self": int(xp_returns.size),
    }


def print_summary(summary: Dict):
    print(
        "Average SP performance "
        f"(diagonal/self-copy): {summary['average_sp']:.3f} "
        f"+- {summary['standard_error_sp']:.3f} SE "
        f"over {summary['num_sp_pairs']} pairs"
    )
    print(
        "Average XP performance "
        f"(off-diagonal/excluding self): {summary['average_xp_excluding_self']:.3f} "
        f"+- {summary['standard_error_xp_excluding_self']:.3f} SE "
        f"over {summary['num_xp_pairs_excluding_self']} ordered pairs"
    )


def save_summary(summary: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("metric,value\n")
        for key, value in summary.items():
            f.write(f"{key},{value}\n")
    print(f"Saved summary to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SP/XP average and standard error from a saved xp_matrix.csv."
    )
    parser.add_argument(
        "xp_matrix_csv",
        help="Path to a saved xp_matrix.csv file.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Where to write the summary CSV. Defaults to summary_with_se.csv next to xp_matrix_csv.",
    )
    parser.add_argument(
        "--overwrite-summary",
        action="store_true",
        help="Write to summary.csv next to xp_matrix_csv unless --summary-csv is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    xp_matrix = np.loadtxt(args.xp_matrix_csv, delimiter=",")
    summary = summarize_xp_matrix(xp_matrix)
    print(f"Loaded XP matrix from {args.xp_matrix_csv}")
    print_summary(summary)

    if args.summary_csv is not None:
        summary_path = args.summary_csv
    elif args.overwrite_summary:
        summary_path = os.path.join(os.path.dirname(args.xp_matrix_csv), "summary.csv")
    else:
        summary_path = os.path.join(os.path.dirname(args.xp_matrix_csv), "summary_with_se.csv")
    save_summary(summary, summary_path)


if __name__ == "__main__":
    main()
