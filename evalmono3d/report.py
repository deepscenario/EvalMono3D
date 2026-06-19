"""Pretty-printing helpers for evaluation results (Markdown-style tables)."""

from __future__ import annotations

import itertools
from typing import Mapping

from tabulate import tabulate

__all__ = ["metrics_table", "per_category_ap_table", "category_histogram"]


def metrics_table(metrics: Mapping[str, float]) -> str:
    """Two-row table of ``{name: value}`` summary metrics."""
    keys, values = zip(*metrics.items())
    return tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        numalign="left",
        stralign="center",
    )


def per_category_ap_table(per_category_ap: Mapping[str, float], n_cols: int = 6) -> str:
    """Compact multi-column table of per-category AP (values already in %)."""
    n_cols = min(n_cols, max(2, len(per_category_ap) * 2))
    flat = list(itertools.chain(*per_category_ap.items()))
    rows = itertools.zip_longest(*[flat[i::n_cols] for i in range(n_cols)])
    return tabulate(
        rows,
        headers=["category", "AP"] * (n_cols // 2),
        tablefmt="pipe",
        floatfmt=".3f",
        numalign="left",
    )


def category_histogram(per_category: Mapping[str, Mapping[str, float]], n_cols: int = 9) -> str:
    """Side-by-side ``category | AP2D | AP3D`` table across all categories."""
    data = list(
        itertools.chain(*[[cat, vals["AP2D"], vals["AP3D"]] for cat, vals in per_category.items()])
    )
    data.extend([None] * (n_cols - (len(data) % n_cols)))
    rows = itertools.zip_longest(*[data[i::n_cols] for i in range(n_cols)])
    return tabulate(
        rows,
        headers=["category", "AP2D", "AP3D"] * (n_cols // 3),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
