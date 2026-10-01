"""Dataset-independent residual scoring and point interpolation."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import Interpolation, Matching


def residuals(query_xyz, query_normals, template_xyz, template_normals, h, config: Matching, threads=1):
    if not np.isfinite(h) or h <= 0 or not len(template_xyz):
        raise ValueError("A nonempty template and positive spacing h are required")
    distance, ids = cKDTree(template_xyz).query(
        query_xyz, k=list(range(1, config.candidate_k + 1)),
        distance_upper_bound=config.radius_h * h, workers=threads)
    valid = np.isfinite(distance) & (ids < len(template_xyz))
    safe_ids = np.where(valid, ids, 0)
    delta = query_xyz[:, None, :] - template_xyz[safe_ids]
    euclidean = np.linalg.norm(delta, axis=-1) / h
    plane = np.abs(np.sum(delta * template_normals[safe_ids], axis=-1)) / h
    normal = 1 - np.abs(np.sum(query_normals[:, None, :] * template_normals[safe_ids], axis=-1))
    euclidean[~valid] = np.inf
    return {"distance": euclidean, "plane": plane, "normal": normal, "search_distance": distance / h}


def anchor_scores(components, config: Matching):
    """Reweight all candidates before selecting the minimum; penalty is absolute."""
    k = config.candidate_k
    costs = (config.distance_weight * components["distance"][:, :k]
             + config.plane_weight * components["plane"][:, :k]
             + config.normal_weight * components["normal"][:, :k])
    costs[~(components["search_distance"][:, :k] < config.radius_h)] = np.inf
    values = costs.min(axis=1)
    unmatched = ~np.isfinite(values)
    values[unmatched] = config.unmatched_penalty
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite anchor scores")
    return values, unmatched


def interpolate(full_xyz, anchor_xyz, values, config: Interpolation, threads=1):
    if len(anchor_xyz) == 0 or len(values) != len(anchor_xyz):
        raise ValueError("Interpolation requires one score per nonempty anchor")
    k = min(config.k, len(anchor_xyz))
    tree = cKDTree(anchor_xyz)
    result = np.empty(len(full_xyz), dtype=np.float64)
    for start in range(0, len(full_xyz), config.chunk_size):
        stop = min(start + config.chunk_size, len(full_xyz))
        distance, ids = tree.query(full_xyz[start:stop], k=list(range(1, k + 1)), workers=threads)
        if config.power == 0:
            # Preserve the arithmetic used in the archived k=128 tuning runs.
            result[start:stop] = np.cumsum(values[ids], axis=1)[:, -1] / k
        else:
            weights = 1 / np.maximum(distance, config.epsilon) if config.power == 1 else np.maximum(distance, config.epsilon) ** (-config.power)
            weights /= weights.sum(axis=1, keepdims=True)
            result[start:stop] = (values[ids] * weights).sum(axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite interpolated scores")
    return result


def object_score(scores, top_fraction=.01):
    count = max(1, int(np.ceil(top_fraction * len(scores))))
    return float(np.partition(scores, -count)[-count:].mean())
