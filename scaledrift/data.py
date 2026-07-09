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


def iter_parent_maps(field_dir, n_parents, column="kappa", seed=0, max_shards=None):
    """Yield up to ``n_parents`` full convergence maps as float64 arrays.

    Maps are drawn from the first ``max_shards`` shards (round-robin over shards for
    physical diversity). Deterministic given ``seed``.
    """
    paths = shard_paths(field_dir)
    if max_shards is not None:
        paths = paths[:max_shards]
    rng = np.random.default_rng(seed)
    got = 0
    # Pull a few rows from each shard in turn until we have enough parents.
    per_shard = int(np.ceil(n_parents / len(paths)))
    for path in paths:
        col = _read_shard(path, column)
        n = len(col)
        take = min(per_shard, n, n_parents - got)
        idx = rng.choice(n, size=take, replace=False)
        for i in sorted(idx.tolist()):
            yield np.asarray(col[i].as_py(), dtype=np.float64)
            got += 1
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
        t = tile_map(kmap, tile)
        if t:
            parents.append(t)
    return parents
