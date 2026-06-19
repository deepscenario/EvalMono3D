"""End-to-end evaluation: load ground truth + predictions, compute AP, report.

Public entry point: :func:`evaluate`. It mirrors the CDrone protocol used in the
paper -- 2D AP over ``IoU = 0.50:0.95`` and 3D AP at ``IoU = 0.50`` with full
``SO(3)`` rotation -- and returns a plain results dict while logging readable
tables to the console and to ``<log_dir>/log.txt``.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import logging
import math
import os
from typing import Any, Iterable, Optional, Sequence

import torch

from .cocoeval import COCOEval3D
from .dataset import DEFAULT_FILTER_SETTINGS, GroundTruth
from .report import category_histogram, metrics_table, per_category_ap_table

logger = logging.getLogger("evalmono3d")

__all__ = ["evaluate", "Omni3DEvaluator"]

Results = dict[str, Any]

# Index i of ``COCOEval3D.stats`` maps to these metric names per mode.
_METRIC_NAMES = {
    "2D": ["AP", "APs", "APm", "APl"],  # area: all / small / medium / large
    "3D": ["AP", "APn", "APm", "APf"],  # depth: all / near / medium / far
}

# Per-instance keys the evaluator relies on (3D keys only when scoring 3D).
_REQUIRED_INSTANCE_KEYS_2D = ("image_id", "category_id", "bbox", "score")
_REQUIRED_INSTANCE_KEYS_3D = ("bbox3D", "depth")


def configure_logging(log_dir: Optional[str] = None) -> None:
    """Attach console (and optional file) handlers to the package logger."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m/%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def _validate_predictions(
    predictions: Sequence[Any], gt_image_ids: Iterable[int], only_2d: bool
) -> None:
    """Fail early, with actionable messages, on malformed prediction dumps."""
    if not isinstance(predictions, (list, tuple)):
        raise TypeError(f"predictions must be a list of per-image records, got {type(predictions).__name__}")
    if len(predictions) == 0:
        raise ValueError("predictions are empty: nothing to evaluate")

    required = _REQUIRED_INSTANCE_KEYS_2D + ((() if only_2d else _REQUIRED_INSTANCE_KEYS_3D))
    gt_ids = set(gt_image_ids)

    for r, record in enumerate(predictions):
        if not isinstance(record, dict) or "instances" not in record:
            raise ValueError(f"prediction record {r} must be a dict with an 'instances' field")
        for instance in record["instances"]:
            missing = [k for k in required if k not in instance]
            if missing:
                raise ValueError(
                    f"prediction instance in record {r} is missing required key(s) {missing}; "
                    f"see DATA.md for the expected schema"
                )
            if instance["image_id"] not in gt_ids:
                raise ValueError(
                    f"prediction references image_id {instance['image_id']!r}, which is not in the "
                    f"ground truth; predictions and ground truth must share image ids"
                )


class Omni3DEvaluator:
    """Computes 2D/3D average precision for one dataset split."""

    def __init__(
        self,
        gt_json_path: str,
        category_names: Sequence[str],
        filter_settings: dict[str, Any],
        only_2d: bool = False,
        iou_3d_thr_range: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.category_names = list(category_names)
        self.only_2d = only_2d
        self.iou_3d_thr_range = iou_3d_thr_range
        self._gt = GroundTruth(gt_json_path, filter_settings)
        self._predictions: list[Any] = []

    def add_predictions(self, predictions: Sequence[Any]) -> None:
        """Append per-image prediction records (see ``DATA.md`` for the schema)."""
        self._predictions += list(predictions)

    def _derive(self, coco_eval: COCOEval3D, mode: str) -> Results:
        """Turn a finished :class:`COCOEval3D` into a ``{metric: value}`` dict (in %)."""
        stats = coco_eval.stats
        results: Results = {
            name: float(stats[i] * 100 if stats[i] >= 0 else math.nan)
            for i, name in enumerate(_METRIC_NAMES[mode])
        }

        # Per-category AP at area/depth "all", 100 detections/image.
        precisions = coco_eval.eval["precision"]
        assert len(self.category_names) == precisions.shape[2]
        per_category: dict[str, float] = {}
        for idx, name in enumerate(self.category_names):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = float(precision.mean() * 100) if precision.size else math.nan
            per_category[name] = ap
            results[f"AP-{name}"] = ap

        present = {n: ap for n, ap in per_category.items() if not math.isnan(ap)}
        if len(present) > 1:
            logger.info("Per-category AP (%s):\n%s", mode, per_category_ap_table(per_category))
        return results

    def evaluate(self, label: str = "-") -> Results:
        """Run evaluation and return ``{"AP2D", "AP3D", "per_category", ...}``.

        Logs the full AP/AR breakdown and a per-category histogram along the way.
        """
        _validate_predictions(self._predictions, self._gt.getImgIds(), self.only_2d)
        instances = list(itertools.chain(*[p["instances"] for p in self._predictions]))
        # In a single-dataset evaluation every predicted category belongs to the
        # dataset and prediction ids are already the contiguous category ids, so
        # no id remapping or vocabulary filtering is required.
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self._gt.loadRes(instances)

        modes = ["2D"] if self.only_2d else ["2D", "3D"]
        results: dict[str, Results] = {}
        for mode in modes:
            coco_eval = COCOEval3D(self._gt, coco_dt, mode=mode, iou_thr_range=self.iou_3d_thr_range)
            coco_eval.evaluate()
            coco_eval.accumulate()
            log_str = coco_eval.summarize()
            logger.info("\n%s", log_str.replace("mode=" + mode, f"{label} mode={mode}"))
            results[mode] = self._derive(coco_eval, mode)

        # Headline numbers: mean AP over the categories present in this split.
        present = [c for c in self.category_names if not math.isnan(results["2D"].get(f"AP-{c}", math.nan))]
        ap2d = float(sum(results["2D"][f"AP-{c}"] for c in present) / len(present)) if present else math.nan
        ap3d = math.nan
        if not self.only_2d:
            ap3d = float(sum(results["3D"][f"AP-{c}"] for c in present) / len(present)) if present else math.nan

        summary = {"AP2D": ap2d, "AP3D": ap3d}
        if not self.only_2d:
            summary.update({"AP3D-N": results["3D"]["APn"], "AP3D-M": results["3D"]["APm"], "AP3D-F": results["3D"]["APf"]})
        logger.info("Overall performance:\n%s", metrics_table(summary))

        per_category = {
            c: {"AP2D": results["2D"][f"AP-{c}"], "AP3D": results["3D"][f"AP-{c}"] if not self.only_2d else math.nan}
            for c in self.category_names
            if not (math.isnan(results["2D"].get(f"AP-{c}", math.nan)))
        }
        logger.info("Per-category performance:\n%s", category_histogram(per_category))

        return {"AP2D": ap2d, "AP3D": ap3d, "per_category": per_category, "raw": results}


def evaluate(
    name: str,
    gt_ann_path: str,
    pred_ann_path: str,
    log_dir: Optional[str] = None,
    only_2d: bool = False,
    iou_3d_thr_range: tuple[float, float] = (0.5, 0.5),
) -> Results:
    """Evaluate predictions against ground truth and return a results dict.

    Args:
        name: label for this split (used in log lines only).
        gt_ann_path: Omni3D-format ground-truth JSON (see ``DATA.md``).
        pred_ann_path: ``.pth`` list of per-image prediction records.
        log_dir: optional directory for ``log.txt``; ``None`` logs to console only.
        only_2d: skip 3D evaluation if ``True``.
        iou_3d_thr_range: ``(lo, hi)`` 3D IoU sweep; CDrone uses ``(0.5, 0.5)``.

    Returns:
        ``{"AP2D", "AP3D", "per_category", "raw"}`` with AP values in percent.
    """
    configure_logging(log_dir)

    with open(gt_ann_path, "r") as f:
        gt = json.load(f)
    category_names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    filter_settings = {**DEFAULT_FILTER_SETTINGS, "category_names": category_names}

    evaluator = Omni3DEvaluator(gt_ann_path, category_names, filter_settings, only_2d, iou_3d_thr_range)
    evaluator.add_predictions(torch.load(pred_ann_path))
    return evaluator.evaluate(label=name)
