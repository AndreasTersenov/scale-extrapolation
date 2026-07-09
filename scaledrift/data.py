"""Loading real fields from the challenge data (HuggingFace Arrow shards).

Each field class lives at ``/project/rrg-lplevass/shared/wl_chall_data/<name>/`` as
``data-*.arrow`` shards with a ``kappa`` column of (1424, 176) convergence maps. We cut
non-overlapping square tiles (default 128^2, the project's train resolution) so the DWT
has clean power-of-two octaves, and keep tiles grouped by PARENT map so the drift
bootstrap resamples the independent unit (see :func:`scaledrift.drift.collect_wc_grouped`).
"""
from __future__ import annotations

import glob
import os

import numpy as np

DATA_ROOT = "/project/rrg-lplevass/shared/wl_chall_data"


def shard_paths(field_dir):
    paths = sorted(glob.glob(os.path.join(field_dir, "data-*.arrow")))
    if not paths:
        raise FileNotFoundError(f"no data-*.arrow shards under {field_dir}")
    return paths


def _read_shard(path, column="kappa"):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(path, "r") as src:
        try:
            reader = ipc.open_stream(src)
        except pa.lib.ArrowInvalid:
            src.seek(0)
            reader = ipc.open_file(src)
        table = reader.read_all()
    return table.column(column)


def iter_parent_maps(field_dir, n_parents, column="kappa", seed=0, max_shards=None,
                     per_shard=None):
    """Yield up to ``n_parents`` FINITE convergence maps as float64 arrays.

    Maps are drawn round-robin across the first ``max_shards`` shards for physical
    diversity. Some entries in these datasets are entirely NaN (masked/blank); those
    are skipped, and shards are revisited (with fresh random rows) until ``n_parents``
    finite maps are found or the pool is exhausted. Deterministic given ``seed``.
    """
    paths = shard_paths(field_dir)
    if max_shards is not None:
        paths = paths[:max_shards]
    rng = np.random.default_rng(seed)
    if per_shard is None:
        per_shard = int(np.ceil(n_parents / len(paths)))
    got = 0
    seen = {p: set() for p in paths}
    for _ in range(64):                       # bounded revisits
        for path in paths:
            col = _read_shard(path, column)
            n = len(col)
            avail = [i for i in range(n) if i not in seen[path]]
            if not avail:
                continue
            take = min(per_shard, len(avail), n_parents - got)
            idx = rng.choice(avail, size=take, replace=False)
            for i in sorted(idx.tolist()):
                seen[path].add(i)
                m = np.asarray(col[i].as_py(), dtype=np.float64)
                if not np.any(np.isfinite(m)):
                    continue                  # skip fully-masked maps (tiles filtered below)
                yield m
                got += 1
                if got >= n_parents:
                    return
        if got >= n_parents:
            return


def tile_map(kmap, tile=128, stride=None):
    """Cut non-overlapping ``tile`` x ``tile`` patches (row-major, cropping remainder)."""
    if stride is None:
        stride = tile
    H, W = kmap.shape
    tiles = []
    for r in range(0, H - tile + 1, stride):
        for c in range(0, W - tile + 1, stride):
            tiles.append(kmap[r:r + tile, c:c + tile])
    return tiles


def load_parent_tiles(field_name, n_parents, tile=128, seed=0, max_shards=8,
                      column="kappa", root=DATA_ROOT):
    """Return ``[[tile, ...], ...]`` -- one inner list of square tiles per parent map.

    Feed straight into :func:`scaledrift.drift.collect_wc_grouped`.
    """
    field_dir = os.path.join(root, field_name)
    parents = []
    for kmap in iter_parent_maps(field_dir, n_parents, column, seed, max_shards):
        t = [x for x in tile_map(kmap, tile) if np.all(np.isfinite(x))]
        if t:
            parents.append(t)
    return parents
