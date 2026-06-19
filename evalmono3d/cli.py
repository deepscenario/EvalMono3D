"""Command-line interface for EvalMono3D.

Installed as the ``evalmono3d`` console script; also reachable via ``eval.py``.

Example::

    evalmono3d \\
        --name cdrone_test_example \\
        --gt_ann  example/cdrone_test_example.json \\
        --pred_ann example/cdrone_test_example_pred.pth \\
        --log_dir /tmp/eval_cdrone_test_example
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .evaluator import evaluate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evalmono3d",
        description="Monocular 3D detection evaluation with full SO(3) IoU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name", required=True, help="label for this split (used in logs)")
    parser.add_argument("--gt_ann", required=True, dest="gt_ann_path", help="ground-truth JSON (Omni3D format)")
    parser.add_argument("--pred_ann", required=True, dest="pred_ann_path", help="predictions .pth file")
    parser.add_argument("--log_dir", default=None, help="directory for log.txt (optional)")
    parser.add_argument("--only_2d", action="store_true", help="skip 3D evaluation")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    evaluate(**vars(parse_args(argv)))


if __name__ == "__main__":
    main()
