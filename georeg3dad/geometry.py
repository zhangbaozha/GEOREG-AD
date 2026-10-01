"""Shared feature extraction, registration, template construction and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import zlib

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .config import Method
from .scoring import anchor_scores, interpolate, object_score, residuals


def transform_xyz(xyz, transform):
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def registration_seed(canonical_filename):
    # Never hash an absolute path or native Windows separators.
    name = str(canonical_filename).replace("\\", "/").rsplit("/", 1)[-1]
    return zlib.crc32(f"test/{name}".encode()) & 0x7fffffff


def read_geometry(path, center=False):
    """Read geometry only: TXT column four must never enter inference."""
    if Path(path).suffix.lower() == '.txt':
        xyz = np.genfromtxt(path, usecols=(0, 1, 2), dtype=np.float64, ndmin=2)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points() or not np.isfinite(np.asarray(cloud.points)).all():
        raise ValueError(f"Empty or invalid point cloud: {path}")
    if center:
        xyz = np.asarray(cloud.points)
        cloud.points = o3d.utility.Vector3dVector(xyz - np.average(xyz, axis=0))
    return cloud


def make_features(path, config, center=False):
    cloud = read_geometry(path, center)
    down = cloud.voxel_down_sample(config.voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=config.normal_radius * config.voxel, max_nn=config.normal_neighbors))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.fpfh_radius * config.voxel, max_nn=config.fpfh_neighbors))
    return cloud, down, feature


def register(source, source_feature, target, target_feature, voxel, config):
    reg = o3d.pipelines.registration
    start = time.perf_counter()
    fallback = None
    try:
        coarse = reg.registration_fgr_based_on_feature_matching(
            source, target, source_feature, target_feature,
            reg.FastGlobalRegistrationOption(maximum_correspondence_distance=config.fgr_distance * voxel,
                                             iteration_number=config.fgr_iterations))
        initial, fitness = coarse.transformation, float(coarse.fitness)
    except RuntimeError as exc:
        # Preserve historical identity fallback, but make it visible in the output.
        initial, fitness, fallback = np.eye(4), 0.0, str(exc)
    fine = reg.registration_icp(source, target, config.icp_distance * voxel, initial,
                               reg.TransformationEstimationPointToPlane(),
                               reg.ICPConvergenceCriteria(relative_fitness=config.relative_fitness,
                                   relative_rmse=config.relative_rmse, max_iteration=config.icp_iterations))
    if not np.isfinite(fine.transformation).all():
        raise ValueError("Registration returned a nonfinite transform")
    return fine.transformation, {"coarse_fitness": fitness, "icp_fitness": float(fine.fitness),
        "icp_inlier_rmse": float(fine.inlier_rmse), "fallback": fallback,
        "seconds": time.perf_counter() - start}


@dataclass
class TemplateLibrary:
    xyz: np.ndarray
    normals: np.ndarray
    h: float
    reference: object
    feature: object
    records: list


class GeoReg3DAD:
    def __init__(self, config: Method, threads=1, center=False):
        self.config = config
        self.threads = threads
        self.center = center
        self.library = None

    def prepare(self, paths):
        cfg = self.config
        paths = sorted(map(Path, paths))[:cfg.templates.count]
        if len(paths) != cfg.templates.count:
            raise ValueError("Not enough normal templates")
        reference_path = paths.pop(cfg.templates.reference_index)
        paths.insert(0, reference_path)
        o3d.utility.random.seed(cfg.templates.seed)
        _, reference, feature = make_features(paths[0], cfg.features, self.center)
        xyz = np.asarray(reference.points)
        if len(xyz) < 2:
            raise ValueError("Reference template needs at least two anchors")
        spacing = cKDTree(xyz).query(xyz, k=2, workers=self.threads)[0][:, 1]
        h = float(np.median(spacing) if cfg.templates.spacing_quantile == .5
                  else np.quantile(spacing, cfg.templates.spacing_quantile))
        if not np.isfinite(h) or h <= 0:
            raise ValueError("Invalid template spacing")
        points, normals = [xyz], [np.asarray(reference.normals)]
        records = [{"template": str(paths[0]), "transformation": np.eye(4).tolist(), "reference": True}]
        for path in paths[1:]:
            _, down, feat = make_features(path, cfg.features, self.center)
            transform, info = register(down, feat, reference, feature, cfg.features.voxel, cfg.registration)
            points.append(transform_xyz(np.asarray(down.points), transform))
            normals.append(np.asarray(down.normals) @ transform[:3, :3].T)
            records.append({"template": str(path), "transformation": transform.tolist(), **info})
        self.library = TemplateLibrary(np.concatenate(points), np.concatenate(normals), h, reference, feature, records)
        return self.library

    def predict(self, path, canonical_filename=None, transform=None, return_intermediates=False):
        if self.library is None:
            raise RuntimeError("Call prepare() before predict()")
        begin = time.perf_counter()
        cfg, lib = self.config, self.library
        seed = registration_seed(canonical_filename or Path(path).name)
        o3d.utility.random.seed(seed)
        full, down, feature = make_features(path, cfg.features, self.center)
        if transform is None:
            transform, info = register(down, feature, lib.reference, lib.feature, cfg.features.voxel, cfg.registration)
        else:
            transform = np.asarray(transform, dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Replay transform must be a finite 4x4 matrix")
            info = {"replayed": True}
        xyz = np.asarray(full.points)
        query = transform_xyz(np.asarray(down.points), transform)
        normals = np.asarray(down.normals) @ transform[:3, :3].T
        components = residuals(query, normals, lib.xyz, lib.normals, lib.h, cfg.matching, self.threads)
        anchors, unmatched = anchor_scores(components, cfg.matching)
        scores = interpolate(transform_xyz(xyz, transform), query, anchors, cfg.interpolation, self.threads)
        record = {"registration_seed": seed, "transformation": transform.tolist(), "registration": info,
                  "full_points": len(xyz), "anchor_points": len(query), "template_points": len(lib.xyz),
                  "spacing_h": lib.h, "effective_k": min(cfg.interpolation.k, len(query)),
                  "unmatched_fraction": float(unmatched.mean()),
                  "object_score": object_score(scores, cfg.object_top_fraction), "seconds": time.perf_counter() - begin}
        if return_intermediates:
            return scores, xyz, record, {'anchor_xyz': np.asarray(down.points),
                'registered_anchors': query, 'raw_scores': anchors,
                'registered_xyz': transform_xyz(xyz, transform)}
        return scores, xyz, record
