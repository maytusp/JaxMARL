#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ZSC_METHODS = (
    "ph2v5",
    "ph2v5_ablate",
    "ph2v4",
    "ph2v4_ablate",
    "e3t",
    "sp",
    "lmpred_ema",
    "lmpred_ema_ablate",
    "lmpred_ema_gamma0",
    "lmpred_ema_gamma09",
    "lmpred_ema_no_self_pred",
)
AD_HOC_METHODS = (
    "ph2v5_ego_ad_hoc_teamplay",
    "ph2v5_ablate_ad_hoc_teamplay",
    "ph2v4_ego_ad_hoc_teamplay",
    "ph2v4_ablate_ad_hoc_teamplay",
    "e3t_ad_hoc_teamplay",
    "sp_ad_hoc_teamplay",
    "fcp_ad_hoc_teamplay",
    "mep_br_ad_hoc_teamplay",
    "pbt_ad_hoc_teamplay",
    "lmpred_ema_ad_hoc_teamplay",
    "lmpred_ema_ablate_ad_hoc_teamplay",
    "lmpred_ema_gamma0_ad_hoc_teamplay",
    "lmpred_ema_gamma09_ad_hoc_teamplay",
    "lmpred_ema_no_self_pred_ad_hoc_teamplay",
)
DEFAULT_LAYOUTS = ("coord_ring", "counter_circuit", "cramped_room5x5")


def read_metric_csv(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("value", "")
            try:
                metrics[row["metric"]] = float(value)
            except ValueError:
                metrics[row["metric"]] = math.nan
    return metrics


def finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def format_mean_se(
    metrics: Dict[str, float],
    mean_key: str,
    se_key: str,
    precision: int,
) -> str:
    mean = metrics.get(mean_key)
    se = metrics.get(se_key)
    if not finite(mean):
        return ""
    if not finite(se):
        return f"{mean:.{precision}f} \\pm nan"
    return f"{mean:.{precision}f} \\pm {se:.{precision}f}"


def display_name(method: str) -> str:
    return method.removesuffix("_ad_hoc_teamplay").removesuffix("_ego")


def discover_layouts(root: Path, methods: Sequence[str]) -> List[str]:
    layouts = set()
    for method in methods:
        method_dir = root / method
        if not method_dir.is_dir():
            continue
        layouts.update(path.name for path in method_dir.iterdir() if path.is_dir())
    return sorted(layouts)


def discover_methods(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def collect_zsc_rows(root: Path, methods: Sequence[str], layouts: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for method in methods:
        row = {"method": display_name(method)}
        has_value = False
        for layout in layouts:
            summary_path = root / method / layout / "summary.csv"
            if not summary_path.exists():
                row[f"{layout} SP"] = ""
                row[f"{layout} XP"] = ""
                continue

            metrics = read_metric_csv(summary_path)
            sp = format_mean_se(metrics, "average_sp", "standard_error_sp", 0)
            xp = format_mean_se(
                metrics,
                "average_xp_excluding_self",
                "standard_error_xp_excluding_self",
                0,
            )
            row[f"{layout} SP"] = sp
            row[f"{layout} XP"] = xp
            has_value = has_value or bool(sp or xp)
        if has_value:
            rows.append(row)
    return rows


def collect_ad_hoc_rows(
    root: Path,
    methods: Sequence[str],
    layouts: Sequence[str],
    filename: str,
    mean_key: str,
    se_key: str,
    precision: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for method in methods:
        row = {"method": display_name(method)}
        has_value = False
        for layout in layouts:
            summary_path = root / method / layout / filename
            if not summary_path.exists():
                row[layout] = ""
                continue

            metrics = read_metric_csv(summary_path)
            value = format_mean_se(metrics, mean_key, se_key, precision)
            row[layout] = value
            has_value = has_value or bool(value)
        if has_value:
            rows.append(row)
    return rows


def markdown_cell(header: str, value: str) -> str:
    if header != "method" and "\\pm" in value:
        return f"${value}$"
    return value


def markdown_table(headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(markdown_cell(header, row.get(header, "")) for header in headers) + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate ZSC and ad-hoc teamplay result tables from summary CSVs."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--zsc-root", type=Path, default=None)
    parser.add_argument("--ad-hoc-root", type=Path, default=None)
    parser.add_argument("--layouts", nargs="*", default=list(DEFAULT_LAYOUTS))
    parser.add_argument("--zsc-methods", nargs="*", default=list(ZSC_METHODS))
    parser.add_argument("--ad-hoc-methods", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    zsc_root = args.zsc_root or repo_root / "zsc_results"
    ad_hoc_root = args.ad_hoc_root or repo_root / "ad_hoc_teamplay_results"
    output_dir = args.output_dir or repo_root / "visualise_tables"
    output_md = args.output_md or output_dir / "tables.md"

    zsc_methods = args.zsc_methods
    ad_hoc_methods = args.ad_hoc_methods or [
        method for method in AD_HOC_METHODS if (ad_hoc_root / method).is_dir()
    ]
    if not ad_hoc_methods:
        ad_hoc_methods = discover_methods(ad_hoc_root)

    layouts = args.layouts or discover_layouts(zsc_root, zsc_methods)

    zsc_headers = ["method"]
    for layout in layouts:
        zsc_headers.extend([f"{layout} SP", f"{layout} XP"])
    ad_hoc_headers = ["method", *layouts]

    zsc_rows = collect_zsc_rows(zsc_root, zsc_methods, layouts)
    ad_hoc_return_rows = collect_ad_hoc_rows(
        ad_hoc_root,
        ad_hoc_methods,
        layouts,
        "summary.csv",
        "average_return",
        "standard_error_return",
        0,
    )
    alignment_rows = collect_ad_hoc_rows(
        ad_hoc_root,
        ad_hoc_methods,
        layouts,
        "alignment_summary.csv",
        "average_alignment_mse",
        "standard_error_alignment_mse",
        4,
    )

    if not zsc_rows:
        raise ValueError(f"No ZSC summaries found under {zsc_root}")
    if not ad_hoc_return_rows:
        raise ValueError(f"No ad-hoc return summaries found under {ad_hoc_root}")
    if not alignment_rows:
        raise ValueError(f"No ad-hoc alignment summaries found under {ad_hoc_root}")

    write_csv(output_dir / "zsc_sp_xp.csv", zsc_headers, zsc_rows)
    write_csv(output_dir / "ad_hoc_return.csv", ad_hoc_headers, ad_hoc_return_rows)
    write_csv(output_dir / "ad_hoc_alignment_mse.csv", ad_hoc_headers, alignment_rows)

    sections = [
        ("ZSC SP/XP Performance", markdown_table(zsc_headers, zsc_rows)),
        ("Ad-Hoc Teamplay Performance", markdown_table(ad_hoc_headers, ad_hoc_return_rows)),
        ("Ad-Hoc Teamplay Alignment MSE", markdown_table(ad_hoc_headers, alignment_rows)),
    ]
    markdown = "\n\n".join(f"## {title}\n\n{table}" for title, table in sections) + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown)

    print(markdown)
    print(f"Wrote tables to {output_md}")
    print(f"Wrote CSV tables to {output_dir}")


if __name__ == "__main__":
    main()
