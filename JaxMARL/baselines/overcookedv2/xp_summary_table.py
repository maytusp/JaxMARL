import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np

from baselines.overcookedv2.eval_xp_from_csv import summarize_xp_matrix


SUMMARY_KEYS = (
    "average_sp",
    "standard_error_sp",
    "average_xp_excluding_self",
    "standard_error_xp_excluding_self",
)


def read_summary_csv(path: str) -> Dict[str, float]:
    summary = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                summary[row["metric"]] = float(row["value"])
            except ValueError:
                summary[row["metric"]] = np.nan
    return summary


def summary_has_se(summary: Dict[str, float]) -> bool:
    return all(key in summary for key in SUMMARY_KEYS)


def load_summary(result_dir: str) -> Optional[Dict[str, float]]:
    summary_path = os.path.join(result_dir, "summary.csv")
    if os.path.exists(summary_path):
        summary = read_summary_csv(summary_path)
        if summary_has_se(summary):
            return summary

    matrix_path = os.path.join(result_dir, "xp_matrix.csv")
    if not os.path.exists(matrix_path):
        return None

    xp_matrix = np.loadtxt(matrix_path, delimiter=",")
    return summarize_xp_matrix(xp_matrix)


def discover_methods(results_root: str) -> List[str]:
    return sorted(
        name
        for name in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, name))
    )


def discover_layouts(results_root: str, methods: List[str]) -> List[str]:
    layouts = set()
    for method in methods:
        method_dir = os.path.join(results_root, method)
        if not os.path.isdir(method_dir):
            continue
        for name in os.listdir(method_dir):
            if os.path.isdir(os.path.join(method_dir, name)):
                layouts.add(name)
    return sorted(layouts)


def format_mean_se(mean: float, se: float, precision: int) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(se):
        return f"{mean:.{precision}f} +- nan"
    return f"{mean:.{precision}f} +- {se:.{precision}f}"


def collect_rows(results_root: str, methods: List[str], layouts: List[str], precision: int):
    rows = []
    for method in methods:
        for layout in layouts:
            result_dir = os.path.join(results_root, method, layout)
            summary = load_summary(result_dir)
            if summary is None:
                continue

            rows.append(
                {
                    "method": method,
                    "layout": layout,
                    "sp_return": summary["average_sp"],
                    "sp_se": summary["standard_error_sp"],
                    "xp_return": summary["average_xp_excluding_self"],
                    "xp_se": summary["standard_error_xp_excluding_self"],
                    "sp_return_pm_se": format_mean_se(
                        summary["average_sp"], summary["standard_error_sp"], precision
                    ),
                    "xp_return_pm_se": format_mean_se(
                        summary["average_xp_excluding_self"],
                        summary["standard_error_xp_excluding_self"],
                        precision,
                    ),
                }
            )
    return rows


def write_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "method",
        "layout",
        "sp_return_pm_se",
        "xp_return_pm_se",
        "sp_return",
        "sp_se",
        "xp_return",
        "xp_se",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV table to {path}")


def make_markdown(rows: List[Dict], methods: List[str], layouts: List[str]) -> str:
    by_key = {(row["method"], row["layout"]): row for row in rows}
    headers = ["method"]
    for layout in layouts:
        headers.extend([f"{layout} SP", f"{layout} XP"])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method in methods:
        values = [method]
        has_any = False
        for layout in layouts:
            row = by_key.get((method, layout))
            if row is None:
                values.extend(["", ""])
                continue
            has_any = True
            values.extend([row["sp_return_pm_se"], row["xp_return_pm_se"]])
        if has_any:
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(markdown: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(markdown)
    print(f"Saved Markdown table to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an SP/XP return +- standard error table across methods and layouts."
    )
    parser.add_argument(
        "--results-root",
        default="xp_results",
        help="Root directory containing <method>/<layout>/summary.csv and xp_matrix.csv files.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Methods to include. Defaults to all method directories under --results-root.",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=None,
        help="Layouts to include. Defaults to all layout directories under selected methods.",
    )
    parser.add_argument(
        "--output-csv",
        default="xp_results/xp_summary_table.csv",
        help="Path for the long-form CSV table.",
    )
    parser.add_argument(
        "--output-md",
        default="xp_results/xp_summary_table.md",
        help="Path for the wide Markdown table.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal places for formatted mean +- SE strings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    methods = args.methods if args.methods is not None else discover_methods(args.results_root)
    layouts = args.layouts if args.layouts is not None else discover_layouts(args.results_root, methods)

    rows = collect_rows(args.results_root, methods, layouts, args.precision)
    if not rows:
        raise ValueError(f"No XP summaries found under {args.results_root}")

    write_csv(rows, args.output_csv)
    markdown = make_markdown(rows, methods, layouts)
    write_markdown(markdown, args.output_md)
    print(markdown)


if __name__ == "__main__":
    main()
