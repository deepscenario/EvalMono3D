"""Ground-truth loading and annotation filtering.

The ground truth is stored in the Omni3D / Cube R-CNN JSON schema (see
``DATA.md``). :class:`GroundTruth` is a thin :class:`pycocotools.coco.COCO`
subclass that, while loading, decides for every annotation:

* which 2D box represents it (``area`` and 2D matching),
* whether it should be *ignored* during evaluation (:func:`is_ignored`),

and attaches the 3D box and object depth used by the 3D metric.

A single, well-defined :data:`DEFAULT_FILTER_SETTINGS` reproduces the CDrone
benchmark protocol; pass your own dict to deviate.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from pycocotools.coco import COCO

logger = logging.getLogger("evalmono3d")

__all__ = ["GroundTruth", "is_ignored", "DEFAULT_FILTER_SETTINGS", "xyxy_to_xywh"]

# An annotation / filter-settings record; values are heterogeneous by design.
Annotation = dict[str, Any]
FilterSettings = dict[str, Any]
Box = list[float]


# CDrone evaluation protocol. Objects that are heavily truncated, barely
# visible, too small / too large in the image, or geometrically unreliable do
# not count for or against a detector.
DEFAULT_FILTER_SETTINGS: FilterSettings = {
    "category_names": None,  # filled in from the ground-truth categories
    "truncation_thres": 1 / 3,
    "visibility_thres": 1 / 3,
    "min_height_thres": 0.02,  # fraction of image height
    "max_height_thres": 1.5,  # fraction of image height
    "max_depth": 1e5,
    "modal_2D_boxes": False,  # use amodal (full-extent) 2D boxes
    "trunc_2D_boxes": True,  # prefer image-truncated projected boxes
}


def xyxy_to_xywh(box: Box) -> Box:
    """Convert an ``[x1, y1, x2, y2]`` box to ``[x, y, w, h]`` (type preserved)."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _select_2d_box(anno: Annotation, filter_settings: FilterSettings) -> Optional[Box]:
    """Pick the best available 2D box as ``[x, y, w, h]``, or ``None``.

    Preference order mirrors the original protocol: truncated projection,
    then full projection, then the tight box.
    """
    trunc = anno.get("bbox2D_trunc")
    if filter_settings["trunc_2D_boxes"] and trunc is not None and not np.all([v == -1 for v in trunc]):
        return xyxy_to_xywh(trunc)
    if anno["bbox2D_proj"][0] != -1:
        return xyxy_to_xywh(anno["bbox2D_proj"])
    if anno["bbox2D_tight"][0] != -1:
        return xyxy_to_xywh(anno["bbox2D_tight"])
    return None


def is_ignored(anno: Annotation, filter_settings: FilterSettings, image_height: int) -> bool:
    """Whether ``anno`` should be ignored (neither a positive nor a hard negative).

    An ignored ground truth can still be matched by a detection (suppressing a
    false positive) but is never required to be found.
    """
    if anno["behind_camera"] or not bool(anno["valid3D"]):
        return True

    # Degenerate or out-of-range 3D geometry.
    if min(anno["dimensions"]) <= 0:
        return True
    if anno["center_cam"][2] > filter_settings["max_depth"]:
        return True
    if anno["lidar_pts"] == 0 or anno["segmentation_pts"] == 0 or anno["depth_error"] > 0.5:
        return True

    # 2D box used for the height-in-image test.
    if filter_settings["modal_2D_boxes"] and anno.get("bbox2D_tight", [-1])[0] != -1:
        box = xyxy_to_xywh(anno["bbox2D_tight"])
    elif (
        filter_settings["trunc_2D_boxes"]
        and "bbox2D_trunc" in anno
        and not np.all([v == -1 for v in anno["bbox2D_trunc"]])
    ):
        box = xyxy_to_xywh(anno["bbox2D_trunc"])
    elif "bbox2D_proj" in anno:
        box = xyxy_to_xywh(anno["bbox2D_proj"])
    else:
        box = anno["bbox"]

    height = box[3]
    if height <= filter_settings["min_height_thres"] * image_height:
        return True
    if height >= filter_settings["max_height_thres"] * image_height:
        return True

    if anno["truncation"] >= 0 and anno["truncation"] >= filter_settings["truncation_thres"]:
        return True
    if anno["visibility"] >= 0 and anno["visibility"] <= filter_settings["visibility_thres"]:
        return True

    return False


class GroundTruth(COCO):
    """COCO-style ground truth with Omni3D 3D annotations and ignore flags."""

    def __init__(self, annotation_file: str, filter_settings: FilterSettings) -> None:
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        logger.info("Loading annotations from %s ...", annotation_file)
        tic = time.time()
        with open(annotation_file, "r") as f:
            dataset = json.load(f)
        assert isinstance(dataset, dict), f"unsupported annotation format: {type(dataset)}"
        logger.info("Done (t=%.2fs)", time.time() - tic)
        self.dataset = dataset

        category_names = filter_settings["category_names"]
        self.dataset["categories"] = sorted(
            (c for c in dataset["categories"] if c["name"] in category_names),
            key=lambda c: c["id"],
        )

        image_height = {im["id"]: im["height"] for im in dataset["images"]}

        kept = []
        for anno in dataset["annotations"]:
            box2d = _select_2d_box(anno, filter_settings)
            if box2d is None:
                continue

            ignore = is_ignored(anno, filter_settings, image_height[anno["image_id"]])
            anno["area"] = box2d[2] * box2d[3]
            anno["iscrowd"] = False
            anno["ignore"] = anno["ignore2D"] = anno["ignore3D"] = ignore
            anno["bbox"] = box2d
            anno["bbox3D"] = anno["bbox3D_cam"]
            anno["depth"] = anno["center_cam"][2]

            if anno["category_name"] in category_names:
                kept.append(anno)

        self.dataset["annotations"] = kept
        with contextlib.redirect_stdout(io.StringIO()):
            self.createIndex()
