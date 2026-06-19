"""Pairwise 3D Intersection-over-Union for arbitrarily oriented boxes.

This is the heart of the evaluation. Unlike KITTI / Waymo / nuScenes, which
assume objects only rotate about the gravity axis (yaw), drone imagery sees
objects under the *full* ``SO(3)`` rotation group. We therefore compute the IoU
of two general 3D cuboids from their eight corners, using the exact
convex-polyhedron intersection routine shipped with PyTorch3D.

The computation runs on the CPU in ``float32`` so that results are fully
deterministic and independent of the available hardware.
"""

import logging

import torch
import torch.nn.functional as F
from pytorch3d import _C
from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles

logger = logging.getLogger("evalmono3d")

__all__ = ["box3d_overlap"]


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Return a mask that is ``True`` where each box's faces are coplanar."""
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> four (B, P, 3) corner tensors per face.
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Face normal from two in-plane edges.
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # The fourth corner must lie in the same plane.
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    return (mat1.bmm(mat2).abs() < eps).view(B)


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a mask that is ``True`` where each box has non-degenerate faces."""
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> three (B, T, 3) corner tensors per triangle.
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2
    return (face_areas > eps).all(1).view(B)


def box3d_overlap(
    boxes_dt: torch.Tensor,
    boxes_gt: torch.Tensor,
    eps_coplanar: float = 1e-4,
    eps_nonzero: float = 1e-8,
) -> torch.Tensor:
    """Volumetric IoU between every detection and every ground-truth box.

    Args:
        boxes_dt: ``(N, 8, 3)`` corners of the detected boxes.
        boxes_gt: ``(M, 8, 3)`` corners of the ground-truth boxes.

    Both inputs must list their 8 corners in the canonical order::

            (4) +---------+. (5)
                | ` .     |  ` .
                | (0) +---+-----+ (1)
                |     |   |     |
            (7) +-----+---+. (6)|
                ` .   |     ` . |
                (3) ` +---------+ (2)

    Returns:
        ``(N, M)`` tensor of IoU values, ``iou = vol_i / (vol_dt + vol_gt - vol_i)``.

    Detections whose corners are not coplanar or that have zero volume are
    invalid for the intersection solver; their IoU row is forced to zero.
    """
    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]

    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        logger.warning("Skipping %d non-coplanar detection boxes at eval.", int(invalid_coplanar.sum()))

    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        logger.warning("Skipping %d zero-volume detection boxes at eval.", int(invalid_nonzero.sum()))

    return ious
