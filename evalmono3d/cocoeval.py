"""COCO-style average-precision engine for 2D and 3D boxes.

:class:`COCOEval3D` is a focused descendant of :class:`pycocotools.cocoeval.COCOeval`.
The greedy score-ordered matching, 101-point precision interpolation and AP
accumulation are identical to the COCO protocol; the only change is the IoU:

* ``mode="2D"`` -- axis-aligned image-box IoU (``pycocotools``), swept over
  ``IoU = 0.50 : 0.95``, broken down by box **area** (all / small / medium / large).
* ``mode="3D"`` -- full ``SO(3)`` volumetric IoU (:func:`evalmono3d.iou3d.box3d_overlap`),
  evaluated at ``IoU = 0.50``, broken down by object **depth** (all / near / medium / far).

The numerics are deterministic: 3D IoU is computed on the CPU in ``float32`` and
all sorts use a stable ``mergesort``.
"""

from __future__ import annotations

import datetime
import logging
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import pycocotools.mask as mask_utils
import torch
from pycocotools.coco import COCO

from .iou3d import box3d_overlap

logger = logging.getLogger("evalmono3d")

__all__ = ["COCOEval3D", "EvalParams"]


class EvalParams:
    """Evaluation grid (IoU / recall thresholds, detection caps, area bins)."""

    def __init__(self, mode: str = "2D", iou_3d_thr_range: tuple[float, float] = (0.05, 0.5)) -> None:
        if mode not in ("2D", "3D"):
            raise ValueError(f"mode {mode!r} not supported")
        self.mode = mode
        self.iouType = "bbox"
        self.imgIds = []
        self.catIds = []
        self.useCats = 1
        self.maxDets = [1, 10, 100]
        # 101-point recall interpolation, as in COCO.
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)

        if mode == "2D":
            # IoU swept 0.50:0.95; ranges are squared pixel areas.
            self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
            self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
            self.areaRngLbl = ["all", "small", "medium", "large"]
        else:
            lo, hi = iou_3d_thr_range
            self.iouThrs = np.linspace(lo, hi, int(np.round((hi - lo) / 0.05)) + 1, endpoint=True)
            # ranges are object depths in metres.
            self.areaRng = [[0, 1e5], [0, 10], [10, 35], [35, 1e5]]
            self.areaRngLbl = ["all", "near", "medium", "far"]


class COCOEval3D:
    """COCO average precision for either 2D image boxes or 3D ``SO(3)`` boxes."""

    def __init__(
        self,
        cocoGt: COCO,
        cocoDt: COCO,
        mode: str = "2D",
        iou_thr_range: tuple[float, float] = (0.05, 0.5),
    ) -> None:
        if mode not in ("2D", "3D"):
            raise ValueError(f"mode {mode!r} not supported")
        self.mode = mode
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.params = EvalParams(mode, iou_3d_thr_range=iou_thr_range)
        self.params.imgIds = sorted(cocoGt.getImgIds())
        self.params.catIds = sorted(cocoGt.getCatIds())

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.evalImgs = []
        self.eval = {}
        self.stats = []
        self.ious = {}

    # -- per-image evaluation ------------------------------------------------

    def _prepare(self) -> None:
        """Group ground-truth and detections by ``(image, category)``."""
        p = self.params
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

        ignore_flag = "ignore2D" if self.mode == "2D" else "ignore3D"
        for gt in gts:
            gt[ignore_flag] = gt.get(ignore_flag, 0)

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)
        self.eval = {}

    def evaluate(self) -> None:
        """Compute IoUs and per-image matches for every ``(image, category)``."""
        logger.info("Running per-image %s evaluation ...", self.mode)
        tic = time.time()

        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self._prepare()

        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId) for imgId in p.imgIds for catId in p.catIds
        }

        maxDet = p.maxDets[-1]
        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in p.catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        logger.info("Done (t=%.2fs).", time.time() - tic)

    def computeIoU(self, imgId: int, catId: int):
        """IoU matrix between detections and ground truth for one image/category."""
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []

        # Keep the highest-scoring detections (stable order).
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[: p.maxDets[-1]]

        if self.mode == "2D":
            d = [d["bbox"] for d in dt]
            g = [g["bbox"] for g in gt]
            return mask_utils.iou(d, g, [0] * len(gt))

        # 3D: SO(3) volumetric IoU, computed deterministically on the CPU.
        if len(dt) > 0 and len(gt) > 0:
            d = torch.tensor([d["bbox3D"] for d in dt], dtype=torch.float32)
            g = torch.tensor([g["bbox3D"] for g in gt], dtype=torch.float32)
            return box3d_overlap(d, g).numpy()
        return []

    def evaluateImg(self, imgId: int, catId: int, aRng: list[float], maxDet: int) -> Optional[dict]:
        """Greedy score-ordered matching for one image, category and range bin."""
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None

        range_key = "area" if self.mode == "2D" else "depth"
        ignore_flag = "ignore2D" if self.mode == "2D" else "ignore3D"

        for g in gt:
            g["_ignore"] = 1 if (g[ignore_flag] or g[range_key] < aRng[0] or g[range_key] > aRng[1]) else 0

        # Sort ground truth (ignored last) and detections (highest score first).
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[:maxDet]]

        iou_mat = self.ious[imgId, catId]
        ious = iou_mat[:, gtind] if len(iou_mat) > 0 else iou_mat

        T, G, D = len(p.iouThrs), len(gt), len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))

        if len(ious) != 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, _ in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1  # best-matched gt index (-1 == unmatched)
                    for gind, _ in enumerate(gt):
                        if gtm[tind, gind] > 0:  # gt already taken
                            continue
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:  # crossed into ignore gts
                            break
                        if ious[dind, gind] < iou:  # not better than current best
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = dt[dind]["id"]

        # Unmatched detections outside the range bin are ignored.
        out_of_range = np.array([d[range_key] < aRng[0] or d[range_key] > aRng[1] for d in dt]).reshape((1, D))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(out_of_range, T, 0)))

        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    # -- accumulation & summary ---------------------------------------------

    def accumulate(self) -> None:
        """Accumulate per-image matches into precision / recall tensors."""
        logger.info("Accumulating %s results ...", self.mode)
        tic = time.time()
        assert self.evalImgs, "Please run evaluate() first"

        p = self.params
        T, R = len(p.iouThrs), len(p.recThrs)
        K, A, M = len(p.catIds), len(p.areaRng), len(p.maxDets)
        I = len(p.imgIds)
        precision = -np.ones((T, R, K, A, M))  # -1 == category/area absent
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # evalImgs is laid out as [category][area][image]; gather each (cat, area).
        for k in range(K):
            for a in range(A):
                base = (k * A + a) * I
                E = [e for e in self.evalImgs[base : base + I] if e is not None]
                if len(E) == 0:
                    continue

                for m, maxDet in enumerate(p.maxDets):
                    dtScores = np.concatenate([e["dtScores"][:maxDet] for e in E])
                    # mergesort matches the reference COCO/Matlab ordering.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, :maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, :maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue

                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=1).astype(float)
                    fp_sum = np.cumsum(fps, axis=1).astype(float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))
                        recall[t, k, a, m] = rc[-1] if nd else 0

                        # Make precision monotonically decreasing in recall.
                        pr = pr.tolist()
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds_r = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds_r):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except IndexError:
                            pass
                        precision[t, :, k, a, m] = q
                        scores[t, :, k, a, m] = ss

        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        logger.info("Done (t=%.2fs).", time.time() - tic)

    def summarize(self) -> str:
        """Build the human-readable AP/AR summary table and return it as a string."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first")
        p = self.params
        range_word = "area" if self.mode == "2D" else "depth"
        lines = []

        def _mean(ap, iouThr, areaRng, maxDets):
            s = self.eval["precision"] if ap else self.eval["recall"]
            aind = p.areaRngLbl.index(areaRng)
            mind = p.maxDets.index(maxDets)
            if iouThr is not None:
                t = np.where(np.isclose(iouThr, p.iouThrs.astype(float)))[0]
                s = s[t]
            s = s[..., aind, mind]
            valid = s[s > -1]
            return -1.0 if len(valid) == 0 else float(np.mean(valid))

        def _emit(ap, iouThr=None, areaRng="all", maxDets=100):
            title = "Average Precision" if ap else "Average Recall"
            typ = "(AP)" if ap else "(AR)"
            iou = (
                f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}" if iouThr is None else f"{iouThr:0.2f}"
            )
            val = _mean(ap, iouThr, areaRng, maxDets)
            lines.append(
                f"mode={self.mode}  {title:<18} {typ} @[ IoU={iou:<9} | "
                f"{range_word}={areaRng:>6s} | maxDets={maxDets:>3d} ] = {val:0.3f}"
            )
            return val

        stats = np.zeros((10,))
        stats[0] = _emit(ap=1)
        stats[1] = _emit(ap=1, areaRng=p.areaRngLbl[1], maxDets=p.maxDets[2])
        stats[2] = _emit(ap=1, areaRng=p.areaRngLbl[2], maxDets=p.maxDets[2])
        stats[3] = _emit(ap=1, areaRng=p.areaRngLbl[3], maxDets=p.maxDets[2])
        stats[4] = _emit(ap=0, maxDets=p.maxDets[0])
        stats[5] = _emit(ap=0, maxDets=p.maxDets[1])
        stats[6] = _emit(ap=0, maxDets=p.maxDets[2])
        stats[7] = _emit(ap=0, areaRng=p.areaRngLbl[1], maxDets=p.maxDets[2])
        stats[8] = _emit(ap=0, areaRng=p.areaRngLbl[2], maxDets=p.maxDets[2])
        stats[9] = _emit(ap=0, areaRng=p.areaRngLbl[3], maxDets=p.maxDets[2])
        self.stats = stats
        return "\n".join(lines)
