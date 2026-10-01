"""Validated settings; importing this module does not initialize numerical libraries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Features:
    voxel: float = 0.2
    normal_radius: float = 2.5  # multiples of voxel
    normal_neighbors: int = 32
    fpfh_radius: float = 5.0
    fpfh_neighbors: int = 100


@dataclass(frozen=True)
class Registration:
    fgr_distance: float = 2.5  # multiples of voxel
    fgr_iterations: int = 64
    icp_distance: float = 2.5
    icp_iterations: int = 50
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6


@dataclass(frozen=True)
class Matching:
    candidate_k: int = 8
    radius_h: float = 8.0
    distance_weight: float = 1.0
    plane_weight: float = 0.5
    normal_weight: float = 0.5
    unmatched_penalty: float = 16.0


@dataclass(frozen=True)
class Interpolation:
    k: int = 3
    power: float = 1.0
    chunk_size: int = 4096
    epsilon: float = 1e-8


@dataclass(frozen=True)
class Templates:
    count: int = 4
    reference_index: int = 0
    spacing_quantile: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class Method:
    features: Features = Features()
    registration: Registration = Registration()
    matching: Matching = Matching()
    interpolation: Interpolation = Interpolation()
    templates: Templates = Templates()
    object_top_fraction: float = 0.01


SECTIONS = {"features": Features, "registration": Registration, "matching": Matching,
            "interpolation": Interpolation, "templates": Templates}


def positive(value, name, *, zero=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def method_from_dict(data):
    if not isinstance(data, dict) or set(data) - {*SECTIONS, "object_top_fraction"}:
        raise ValueError("Unknown method section")
    sections = {}
    for name, cls in SECTIONS.items():
        values = data.get(name, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown fields in {name}")
        section = cls(**values)
        for f in fields(cls):
            value = getattr(section, f.name)
            positive(value, f"{name}.{f.name}",
                     zero=f.name in {"power", "plane_weight", "normal_weight", "unmatched_penalty", "reference_index", "seed"},
                     integer=isinstance(f.default, int))
        sections[name] = section
    method = Method(**sections, object_top_fraction=data.get("object_top_fraction", .01))
    positive(method.object_top_fraction, "object_top_fraction")
    if method.object_top_fraction > 1:
        raise ValueError("object_top_fraction must be <= 1")
    if method.templates.count > 4 or method.templates.reference_index >= method.templates.count:
        raise ValueError("Select 1..4 templates and a reference index within that subset")
    if method.templates.spacing_quantile > 1 or method.templates.seed > 0x7fffffff:
        raise ValueError("Invalid template spacing quantile or seed")
    if method.features.fpfh_radius <= method.features.normal_radius:
        raise ValueError("FPFH radius must exceed the normal-estimation radius")
    return method


def merge(base, changes):
    if not isinstance(base, dict) or not isinstance(changes, dict):
        raise ValueError("Configuration sections must be objects")
    result = dict(base)
    for key, value in changes.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path, preset=None, overrides=()):
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or set(document) - {"dataset", "description", "method", "presets"}:
        raise ValueError("Unknown top-level configuration field")
    if document.get("dataset") not in {"real3dad", "shapenet"}:
        raise ValueError("dataset must be real3dad or shapenet")
    values = document.get("method", {})
    if preset:
        if preset not in document.get("presets", {}):
            raise ValueError(f"Unknown preset: {preset}")
        values = merge(values, document["presets"][preset])
    for override in overrides:
        key, separator, raw = override.partition("=")
        parts = key.split(".")
        if not separator or len(parts) not in (1, 2):
            raise ValueError("Use --set section.parameter=JSON_value")
        value = json.loads(raw)
        change = {parts[0]: {parts[1]: value}} if len(parts) == 2 else {key: value}
        values = merge(values, change)
    method = method_from_dict(values)
    return {"dataset": document["dataset"], "method": asdict(method),
            "preset": preset or "default", "description": document.get("description", "")}, method
