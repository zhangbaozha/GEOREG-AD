"""Dataset layouts and GT protocols; scoring has no knowledge of either dataset."""
from __future__ import annotations

from pathlib import Path

from .runtime import read_json, sha256


def protocol(dataset):
    if dataset not in {"real3dad", "shapenet"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    return read_json(Path(__file__).parent / "protocols" / f"{dataset}.json")


def canonical_filename(dataset, filename):
    return filename.replace("_positive", "_good") if dataset == "shapenet" else filename


def point_count(path):
    with Path(path).open("rb") as stream:
        for _ in range(100):
            line = stream.readline().decode("ascii").strip()
            if line.startswith("POINTS "):
                count = int(line.split()[1])
                if count <= 0:
                    raise ValueError(f"Empty PCD: {path}")
                return count
            if line.startswith("DATA "):
                break
    raise ValueError(f"PCD POINTS header missing: {path}")


def inspect_dataset(dataset, data_root, categories=(), scope="all", smoke=False,
                    input_protocol="legacy-pcd"):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Data root does not exist: {root}")
    spec = protocol(dataset)
    if input_protocol not in {'legacy-pcd', 'real3dad-official'}:
        raise ValueError('Unknown input protocol')
    official = input_protocol == 'real3dad-official'
    if official and dataset != 'real3dad':
        raise ValueError('Official TXT input is specific to Real3D-AD')
    expected = [r for r in spec['categories'] if scope == "all" or r['source_split'] == scope]
    requested = set(categories)
    if requested - {r['category'] for r in expected}:
        raise ValueError(f"Unknown categories for this scope: {sorted(requested)}")
    if requested:
        expected = [r for r in expected if r['category'] in requested]
    if not expected:
        raise ValueError("No categories selected")
    result = []
    for row in expected:
        folder = root / row['source_category'] if dataset == "real3dad" else root / row['source_split'] / row['source_category']
        templates = sorted((folder / "train").glob("*.pcd"))
        tests = sorted((folder / "test").glob("*.pcd"))
        normal = lambda p: "good" in p.stem if dataset == "real3dad" else "_positive" in p.stem
        actual = {"templates": len(templates), "test_scans": len(tests), "normal_scans": sum(normal(p) for p in tests)}
        if actual != {"templates": 4, "test_scans": row['test_scans'], "normal_scans": row['normal_scans']}:
            raise ValueError(f"Dataset coverage differs for {row['category']}: {actual}")
        gt_dir = folder / ("gt" if dataset == "real3dad" else "GT")
        if len(list(gt_dir.glob('*.txt'))) != row['test_scans'] - row['normal_scans']:
            raise ValueError(f"GT file coverage differs for {row['category']}")
        exclusions = {r['sample']: r for r in spec['exclusions'] if r['category'] == row['category']}
        cases = []
        for test in tests:
            anomaly = not normal(test)
            gt = gt_dir / f"{test.stem}.txt" if anomaly else None
            if gt is not None and not gt.is_file():
                raise ValueError(f"Missing GT: {gt}")
            excluded = exclusions.get(test.stem) if not official else None
            if excluded and (gt is None or sha256(gt) != excluded['sha256']):
                raise ValueError(f"Known excluded GT changed: {gt}; audit it again")
            input_path = gt if official and anomaly else test
            if official and anomaly:
                with gt.open(encoding='utf-8') as stream:
                    points = sum(bool(line.split('#', 1)[0].strip()) for line in stream)
                if not points:
                    raise ValueError(f'Empty GT geometry: {gt}')
            else:
                points = point_count(test)
            cases.append({"test": str(test), "gt": str(gt) if gt else None, "is_anomaly": anomaly,
                          "sample": Path(canonical_filename(dataset, test.name)).stem,
                          "input": str(input_path),
                          "point_gt_valid": not bool(excluded), "points": points})
        if len({c['sample'] for c in cases}) != len(cases):
            raise ValueError("Canonical sample-name collision")
        if sum(c['point_gt_valid'] for c in cases) != (row['test_scans'] if official else row['point_valid_scans']):
            raise ValueError(f"Point-GT coverage differs for {row['category']}")
        if smoke:
            cases = [next(c for c in cases if not c['is_anomaly']),
                     next(c for c in cases if c['is_anomaly'] and c['point_gt_valid'])]
        result.append({**row, "point_valid_scans": sum(c['point_gt_valid'] for c in cases),
                       "templates": [str(p) for p in templates], "cases": cases})
    return {"dataset": dataset, "data_root": str(root), "scope": scope, "smoke": smoke,
            "input_protocol": input_protocol, "center_inputs": official,
            "categories": result, "test_scans": sum(len(r['cases']) for r in result),
            "point_valid_scans": sum(c['point_gt_valid'] for r in result for c in r['cases'])}


def official_labels(case):
    """Called after prediction; TXT labels have the exact geometry row order."""
    import numpy as np
    if case['gt'] is None:
        return np.zeros(case['points'], dtype=np.int8)
    labels = np.genfromtxt(case['gt'], usecols=(3,), dtype=np.float64, ndmin=1)
    if len(labels) != case['points'] or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Invalid official labels: {case['gt']}")
    return labels.astype(np.int8)


def align_labels(xyz, gt_path, dataset, threads=1, tolerance=1e-5):
    import numpy as np
    from scipy.spatial import cKDTree
    with Path(gt_path).open(encoding="utf-8") as stream:
        delimiter = "," if "," in stream.readline() else None
    if dataset == "real3dad" and delimiter is not None:
        raise ValueError("Real3D GT must be whitespace-delimited")
    gt = np.loadtxt(gt_path, delimiter=delimiter, dtype=np.float64, ndmin=2)
    if gt.shape[0] != len(xyz) or gt.shape[1] < 4 or not np.isfinite(gt[:, :4]).all():
        raise ValueError(f"Invalid GT shape/count/values: {gt_path}")
    if not np.isin(gt[:, 3], [0, 1]).all():
        raise ValueError(f"Nonbinary labels: {gt_path}")
    ref, labels = gt[:, :3], gt[:, 3].astype(np.int8)
    aligned = np.empty(len(xyz), dtype=np.int8)
    if delimiter is not None:
        distances, ids = cKDTree(xyz).query(ref, k=1, distance_upper_bound=tolerance, workers=threads)
        if not np.isfinite(distances).all() or len(np.unique(ids)) != len(ids):
            raise ValueError(f"GT must map one-to-one within tolerance: {gt_path}")
        aligned[ids] = labels
        return aligned
    order_xyz = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    order_gt = np.lexsort((ref[:, 2], ref[:, 1], ref[:, 0]))
    error = np.linalg.norm(xyz[order_xyz] - ref[order_gt], axis=1)
    ambiguous = (np.linalg.norm(np.diff(ref[order_gt], axis=0), axis=1) <= tolerance) & (np.diff(labels[order_gt]) != 0)
    if error.max() > tolerance or ambiguous.any():
        raise ValueError(f"GT coordinates differ or duplicate labels conflict: {gt_path}")
    aligned[order_xyz] = labels[order_gt]
    return aligned
