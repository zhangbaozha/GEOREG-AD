"""Exact E=.5/H=2 TFCE on occupied voxels; no tunable CLI parameters.

This is a score enhancement operator, not a permutation significance test.
Extent counts occupied voxels. No presmoothing or per-scan normalization.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def voxel_mapping(full_xyz, anchor_xyz, voxel):
    """Recover the grid used by Open3D voxel_down_sample, preserving anchor order."""
    origin = np.min(full_xyz, axis=0) - voxel * .5
    full_grid = np.floor((full_xyz - origin) / voxel).astype(np.int64)
    grid = np.floor((anchor_xyz - origin) / voxel).astype(np.int64)
    dims = np.max(full_grid, axis=0) + 3
    if float(np.prod(dims.astype(float))) >= np.iinfo(np.int64).max:
        raise ValueError('Grid too large to encode exactly')
    def encode(q):
        return ((q[:, 0] + 1) * dims[1] + q[:, 1] + 1) * dims[2] + q[:, 2] + 1
    keys = encode(grid)
    order = np.argsort(keys)
    ordered = keys[order]
    if len(np.unique(keys)) != len(grid):
        raise ValueError('Multiple anchors in a voxel')
    full_keys = encode(full_grid)
    loc = np.searchsorted(ordered, full_keys)
    if np.any(loc >= len(ordered)) or not np.array_equal(ordered[loc], full_keys):
        raise ValueError('Original point is missing its voxel anchor')
    mapping = order[loc]
    if len(np.unique(mapping)) != len(grid):
        raise ValueError('Unassigned anchor')
    return grid, mapping


def voxel_edges(grid):
    """Undirected 26-connectivity, one copy of each edge; no distance/k search."""
    grid = np.asarray(grid, dtype=np.int64)
    if not len(grid):
        raise ValueError('Empty grid')
    grid = grid - grid.min(axis=0) + 1
    dims = grid.max(axis=0) + 2
    strides = np.array([int(dims[1]) * int(dims[2]), int(dims[2]), 1], dtype=np.int64)
    keys = grid @ strides
    order = np.argsort(keys)
    ordered = keys[order]
    if len(np.unique(keys)) != len(grid):
        raise ValueError('Duplicate grid cells')
    sources, targets = [], []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset <= (0, 0, 0):
            continue
        wanted = keys + np.dot(offset, strides)
        loc = np.searchsorted(ordered, wanted)
        valid = loc < len(ordered)
        ids = np.flatnonzero(valid)
        ids = ids[ordered[loc[ids]] == wanted[ids]]
        sources.append(ids)
        targets.append(order[loc[ids]])
    return np.concatenate(sources), np.concatenate(targets)


def enhance(scores, src, dst):
    """Exact integral int_0^s |component(t)|^.5 * t^2 dt via a merge tree.

    A component born at b and merged at a contributes sqrt(size)*(b^3-a^3)/3
    to each descendant leaf. Equal-score merges have zero interval length,
    making tie results independent of activation order.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if not n or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Finite nonnegative scores required')
    src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    graph = csr_matrix((np.ones(2*len(src), dtype=np.int8),
                        (np.r_[src, dst], np.r_[dst, src])), shape=(n, n))
    parent = np.arange(n)
    active = np.zeros(n, dtype=bool)
    tree_root = np.arange(n)
    extent = np.ones(2*n, dtype=np.int64)
    birth = np.zeros(2*n)
    birth[:n] = scores
    tree_parent = np.full(2*n, -1, dtype=np.int64)
    next_node = n
    def find(a):
        root = a
        while parent[root] != root:
            root = int(parent[root])
        while parent[a] != a:
            previous = int(parent[a]); parent[a] = root; a = previous
        return root
    for value in np.argsort(-scores, kind='stable'):
        i = int(value)
        active[i] = True
        level = scores[i]
        for other in graph.indices[graph.indptr[i]:graph.indptr[i+1]]:
            j = int(other)
            if not active[j]:
                continue
            a, b = find(i), find(j)
            if a == b:
                continue
            ta, tb = int(tree_root[a]), int(tree_root[b])
            if extent[ta] < extent[tb]:
                a, b, ta, tb = b, a, tb, ta
            tree_parent[ta] = tree_parent[tb] = next_node
            birth[next_node] = level
            extent[next_node] = extent[ta] + extent[tb]
            parent[b] = a
            tree_root[a] = next_node
            next_node += 1
    accumulated = np.zeros(next_node)
    for node in range(next_node-1, -1, -1):
        ancestor = int(tree_parent[node])
        lower = birth[ancestor] if ancestor >= 0 else 0.
        accumulated[node] = np.sqrt(extent[node]) * (birth[node]**3 - lower**3) / 3.
        if ancestor >= 0:
            accumulated[node] += accumulated[ancestor]
    result = accumulated[:n]
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Invalid enhanced scores')
    return result


def brute_reference(scores, src, dst):
    """Independent small-graph reference: relabel each unique score interval."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    graph = csr_matrix((np.ones(2*len(src)), (np.r_[src,dst],np.r_[dst,src])), shape=(n,n))
    levels = np.unique(np.r_[0., scores])
    result = np.zeros(n)
    for lo, hi in zip(levels[:-1], levels[1:]):
        active = np.flatnonzero(scores >= hi)
        _, components = connected_components(graph[active][:, active], directed=False)
        sizes = np.bincount(components)
        result[active] += np.sqrt(sizes[components]) * (hi**3-lo**3)/3.
    return result


def self_test():
    rng = np.random.default_rng(20260928)
    max_error = 0.
    for n in (1, 2, 9, 35):
        for repeat in range(12):
            a, b = np.triu_indices(n, k=1)
            keep = rng.random(len(a)) < .15
            a, b = a[keep], b[keep]
            s = rng.integers(0, 5, n).astype(float) if repeat % 2 else rng.uniform(0, 5, n)
            got, ref = enhance(s, a, b), brute_reference(s, a, b)
            max_error = max(max_error, float(np.max(np.abs(got-ref))))
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
            permutation = rng.permutation(n); inverse = np.argsort(permutation)
            permuted = enhance(s[permutation], inverse[a], inverse[b])[inverse]
            np.testing.assert_allclose(got, permuted, rtol=1e-12, atol=1e-12)
    grid = np.array(list(itertools.product(range(3), repeat=3)))
    a, b = voxel_edges(grid)
    actual = {tuple(sorted((int(i),int(j)))) for i,j in zip(a,b)}
    expected = {(i,j) for i in range(len(grid)) for j in range(i+1,len(grid))
                if np.max(np.abs(grid[i]-grid[j])) == 1}
    assert actual == expected
    np.testing.assert_allclose(enhance(np.full(27, 2.),a,b), np.sqrt(27)*8/3)
    np.testing.assert_allclose(enhance(np.array([2., 3.]),np.array([],dtype=int),np.array([],dtype=int)),np.array([8.,27.])/3)
    import open3d as o3d
    points = rng.normal(size=(500,3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    anchors = np.asarray(cloud.voxel_down_sample(.3).points)
    _, assignment = voxel_mapping(points,anchors,.3)
    count = np.bincount(assignment,minlength=len(anchors))
    centroids = np.column_stack([np.bincount(assignment,weights=points[:,j])/count for j in range(3)])
    np.testing.assert_allclose(centroids,anchors,rtol=1e-12,atol=1e-12)
    print(f'TFCE_SELF_TEST_PASS brute_force_max_error={max_error:.3g} ties=PASS permutation=PASS connectivity=PASS voxel_mapping=PASS',flush=True)


if __name__ == '__main__':
    self_test()
