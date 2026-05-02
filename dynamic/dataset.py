"""Dynamic graph datasets.

Two families:
  * OGB node-prop (arxiv / products / papers100M) - dynamic-stream
    protocol: drop edges incident to train_idx, shard into
    contiguous snapshots that re-insert them.
  * SBM-500K (`sbm-500k`) - synthetic SBM with community drift.
    Loads `_init.txt` + `_Edgeupdate_snap{i}.txt` from `data/sbm-500k/`.
    The Edgeupdate files have toggle semantics
    (insert if absent, delete if present); we resolve to (u, v, sigma)
    events by simulating the running edge set.

Use `load_dynamic(dataset=...)` as the unified entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sklearn.preprocessing

# Map our short dataset key -> (OGB name, default number of snapshots)
_OGB_DATASETS = {
    "arxiv": ("ogbn-arxiv", 16),
    "products": ("ogbn-products", 15),
    "papers100M": ("ogbn-papers100M", 20),
}


@dataclass
class DynamicOGB:
    name: str  # dataset key: "arxiv" | "products" | "papers100M"
    features: np.ndarray  # [N, F] float64
    labels: np.ndarray  # [N] int32
    train_idx: np.ndarray  # int32
    val_idx: np.ndarray  # int32
    test_idx: np.ndarray  # int32
    num_nodes: int
    init_edges: np.ndarray  # [E_init, 2] int32, each undirected edge once
    snapshots: List[np.ndarray]  # [M_t, 3] int32 per snapshot, (u, v, sigma=+1)
    label_snapshots: Optional[List[np.ndarray]] = None  # post-snapshot labels


# Backward-compatible alias.
DynamicArxiv = DynamicOGB


def _register_pyg_safe_globals() -> None:
    try:
        from torch.serialization import add_safe_globals
        from torch_geometric.data.data import (
            Data,
            DataEdgeAttr,
            DataTensorAttr,
            EdgeAttr,
            TensorAttr,
        )
        from torch_geometric.data.storage import (
            BaseStorage,
            EdgeStorage,
            GlobalStorage,
            NodeStorage,
        )

        add_safe_globals(
            [
                Data,
                DataEdgeAttr,
                DataTensorAttr,
                EdgeAttr,
                TensorAttr,
                BaseStorage,
                EdgeStorage,
                GlobalStorage,
                NodeStorage,
            ]
        )
    except Exception:
        pass


def load_ogb_dynamic(
    dataset: str = "arxiv",
    root: str = "data/ogb",
    num_snapshots: Optional[int] = None,
    seed: int = 0,
    shuffle_dropped: bool = True,
    drop_mode: str = "train_node",
) -> DynamicOGB:
    """Load an OGB node-prop dataset and produce the dynamic-stream split.

    dataset: "arxiv" | "products" | "papers100M".
    num_snapshots: defaults to the per-dataset convention used in the runs.
    drop_mode:
      - "train_node": drop edges with any endpoint
        in train_idx.
      - "random":     uniform random drop (~20%) of edges (ablation only).
    """
    import torch
    import builtins

    if dataset not in _OGB_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {list(_OGB_DATASETS)}")
    ogb_name, default_snaps = _OGB_DATASETS[dataset]
    if num_snapshots is None:
        num_snapshots = default_snaps

    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_undirected

    _register_pyg_safe_globals()

    import time

    t = time.time()
    _real_torch_load = torch.load
    torch.load = lambda *a, **kw: _real_torch_load(*a, **{**kw, "weights_only": False})
    try:
        ds = PygNodePropPredDataset(name=ogb_name, root=root)
    finally:
        torch.load = _real_torch_load
    data = ds[0]
    print("done loading", time.time() - t)
    split_idx = ds.get_idx_split()
    train_idx = split_idx["train"].numpy().astype(np.int32)
    val_idx = split_idx["valid"].numpy().astype(np.int32)
    test_idx = split_idx["test"].numpy().astype(np.int32)

    t = time.time()
    feat = data.x.numpy().astype(np.float64)
    feat = sklearn.preprocessing.StandardScaler().fit_transform(feat)
    print("done scaling", time.time() - t)

    # papers100M labels are float with NaN entries on nodes outside the split;
    # zero them out so the int cast is well-defined. The MLP only ever reads
    # rows indexed by train/val/test_idx, which exclude those NaN nodes.
    raw_labels = data.y.numpy().reshape(-1)
    if np.issubdtype(raw_labels.dtype, np.floating):
        raw_labels = np.nan_to_num(raw_labels, nan=0.0)
    labels = raw_labels.astype(np.int32)

    num_nodes = int(data.num_nodes)
    # Cast to int32 BEFORE to_undirected so the doubled edge tensor
    # (~3.2B edges on papers100M) is half the size, and so all downstream
    # numpy ops (row/col copies, lo/hi, stack, unique) stay int32. Safe:
    # papers100M has ~111M nodes << INT32_MAX (2.1B).
    t = time.time()
    if dataset == "papers100M":
        row, col = data.edge_index.to(torch.int32).numpy()
        keep = row != col
        pairs = np.stack([row[keep], col[keep]], axis=1)
    else:
        edge_index = to_undirected(data.edge_index.to(torch.int32), num_nodes=num_nodes)
        row, col = edge_index.numpy()
        keep = row != col
        row, col = row[keep], col[keep]
        lo = np.minimum(row, col)
        hi = np.maximum(row, col)
        pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    print("done preparing pairs", time.time() - t)

    if drop_mode == "train_node":
        is_train = np.zeros(num_nodes, dtype=bool)
        is_train[train_idx] = True
        drop_mask = is_train[pairs[:, 0]] | is_train[pairs[:, 1]]
    elif drop_mode == "random":
        rng = np.random.RandomState(seed)
        drop_mask = rng.rand(pairs.shape[0]) < 0.2
    else:
        raise ValueError(f"Unknown drop_mode {drop_mode}")

    drop_edges = pairs[drop_mask]
    init_edges = pairs[~drop_mask]

    if shuffle_dropped:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(drop_edges))
        drop_edges = drop_edges[perm]

    per_snap = len(drop_edges) // num_snapshots
    snapshots: List[np.ndarray] = []
    for sn in range(num_snapshots):
        chunk = drop_edges[sn * per_snap : (sn + 1) * per_snap]
        ev = np.empty((len(chunk), 3), dtype=np.int32)
        ev[:, 0] = chunk[:, 0]
        ev[:, 1] = chunk[:, 1]
        ev[:, 2] = 1
        snapshots.append(ev)
    tail = drop_edges[num_snapshots * per_snap :]
    if len(tail):
        extra = np.empty((len(tail), 3), dtype=np.int32)
        extra[:, 0] = tail[:, 0]
        extra[:, 1] = tail[:, 1]
        extra[:, 2] = 1
        snapshots[-1] = np.concatenate([snapshots[-1], extra], axis=0)

    return DynamicOGB(
        name=dataset,
        features=feat,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_nodes=num_nodes,
        init_edges=init_edges,  # already int32 (propagated from edge_index)
        snapshots=snapshots,
    )


# SBM-500K defaults used by the dynamic runs.
_SBM_DEFAULTS = {
    "sbm-500k": dict(
        stem="SBM-500000-50-20+1",
        num_nodes=500_000,
        num_snapshots=10,
        feat_dim=256,
    ),
}


def _parse_edges_int32(txt_path: str, cache_dir: Optional[str] = None) -> np.ndarray:
    """Parse a whitespace-separated `<u> <v>` edge text file into [E, 2] int32.

    Caches the parsed array next to (or under cache_dir) the txt as
    `<basename>.int32.npy` since the SBM init file is ~14M lines.
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(txt_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, os.path.basename(txt_path) + ".int32.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import pandas as pd

    df = pd.read_csv(txt_path, sep=r"\s+", header=None, dtype=np.int32, engine="c")
    arr = np.ascontiguousarray(df.to_numpy(dtype=np.int32, copy=False))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    np.save(cache, arr)
    return arr


def _undirected_unique_pairs(arr: np.ndarray) -> np.ndarray:
    """[E,2] int32 -> sorted unique unordered pairs (lo<=hi), self-loops dropped."""
    keep = arr[:, 0] != arr[:, 1]
    a = arr[keep]
    lo = np.minimum(a[:, 0], a[:, 1])
    hi = np.maximum(a[:, 0], a[:, 1])
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


def _pair_keys(pairs: np.ndarray) -> np.ndarray:
    """[E,2] int32 (lo<=hi) -> [E] int64 keys = (lo<<32) | hi."""
    return (pairs[:, 0].astype(np.int64) << np.int64(32)) | (
        pairs[:, 1].astype(np.int64) & np.int64(0xFFFFFFFF)
    )


def load_sbm_dynamic(
    dataset: str = "sbm-500k",
    root: str = "data/sbm-500k",
    seed: int = 0,
    feat_dim: Optional[int] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    cache_dir: Optional[str] = None,
) -> DynamicOGB:
    """Load the SBM-500K dynamic dataset.

    Files expected directly under `root` (default `data/sbm-500k`):
      <stem>_init.txt              - initial edge list, both directions per edge
      <stem>_Edgeupdate_snap{i}.txt - per-snapshot toggles (both directions)
      <stem>_label.txt             - initial labels [N], one per line

    Notes:
      * Features are NOT shipped. We synthesize fixed-seed N(0,1) features
        with `feat_dim` columns (default 256 for SBM-500K) and standard-scale
        them.
      * Train/val/test split is a random `train_frac/val_frac/(rest)` partition
        with `seed`.
      * SBM ground-truth labels evolve per snapshot (`<stem>_label_snap{i}.txt`).
        We keep the initial labels in `labels` and post-update label vectors in
        `label_snapshots`, aligned with `snapshots`.
    """
    cfg = _SBM_DEFAULTS.get(dataset.lower())
    if cfg is None:
        raise ValueError(
            f"Unknown SBM dataset {dataset!r}. Known: {list(_SBM_DEFAULTS)}"
        )
    stem = cfg["stem"]
    N = cfg["num_nodes"]
    M = cfg["num_snapshots"]
    F = feat_dim if feat_dim is not None else cfg["feat_dim"]

    init_path = os.path.join(root, f"{stem}_init.txt")
    label_path = os.path.join(root, f"{stem}_label.txt")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"SBM init file missing: {init_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"SBM label file missing: {label_path}")

    cache = cache_dir or os.path.join(root, ".sbm_cache")

    import time

    t = time.time()
    init_raw = _parse_edges_int32(init_path, cache_dir=cache)
    init_pairs = _undirected_unique_pairs(init_raw)
    del init_raw
    print(
        f"loaded SBM init: {init_pairs.shape[0]} unique undirected edges "
        f"({time.time() - t:.1f}s)"
    )

    raw_labels = np.loadtxt(label_path, dtype=np.int32)
    if raw_labels.shape[0] != N:
        raise ValueError(f"SBM label count {raw_labels.shape[0]} != expected N={N}")

    t = time.time()
    feat_rng = np.random.RandomState(seed)

    nnz_per_node = 16
    rows = np.repeat(np.arange(N), nnz_per_node)
    cols = feat_rng.randint(0, F, size=N * nnz_per_node)
    vals = feat_rng.choice([1.0], size=N * nnz_per_node)

    from scipy import sparse as sp

    feat = (
        sp.coo_matrix((vals, (rows, cols)), shape=(N, F)).toarray().astype(np.float64)
    )
    feat = sklearn.preprocessing.StandardScaler().fit_transform(feat)
    print(f"generated random SBM features [{N}, {F}] " f"({time.time() - t:.1f}s)")

    perm = np.random.RandomState(seed + 1).permutation(N).astype(np.int32)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    # Resolve toggle semantics into (u, v, sigma) events by tracking the
    # current edge set as sorted int64 keys.
    current = np.sort(_pair_keys(init_pairs))
    snapshots: List[np.ndarray] = []
    label_snapshots: List[np.ndarray] = []
    for i in range(M):
        upd_path = os.path.join(root, f"{stem}_Edgeupdate_snap{i}.txt")
        snap_label_path = os.path.join(root, f"{stem}_label_snap{i}.txt")
        if not os.path.exists(upd_path):
            raise FileNotFoundError(f"SBM update file missing: {upd_path}")
        if not os.path.exists(snap_label_path):
            raise FileNotFoundError(
                f"SBM snapshot label file missing: {snap_label_path}"
            )
        upd_raw = _parse_edges_int32(upd_path, cache_dir=cache)
        upd_pairs = _undirected_unique_pairs(upd_raw)
        del upd_raw
        upd_keys = _pair_keys(upd_pairs)
        snap_labels = np.loadtxt(snap_label_path, dtype=np.int32)
        if snap_labels.shape[0] != N:
            raise ValueError(
                f"SBM label count {snap_labels.shape[0]} in "
                f"{snap_label_path} != expected N={N}"
            )

        pos = np.searchsorted(current, upd_keys)
        in_set = np.zeros(upd_keys.size, dtype=bool)
        valid = pos < current.size
        in_set[valid] = current[pos[valid]] == upd_keys[valid]
        sigma = np.where(in_set, np.int32(-1), np.int32(1)).astype(np.int32)

        ev = np.empty((upd_keys.size, 3), dtype=np.int32)
        ev[:, 0] = upd_pairs[:, 0]
        ev[:, 1] = upd_pairs[:, 1]
        ev[:, 2] = sigma
        snapshots.append(ev)
        label_snapshots.append(snap_labels)

        deletes = upd_keys[in_set]
        inserts = upd_keys[~in_set]
        if deletes.size:
            keep = np.ones(current.size, dtype=bool)
            keep[np.searchsorted(current, deletes)] = False
            current = current[keep]
        if inserts.size:
            current = np.sort(np.concatenate([current, inserts]))
        n_ins = int((sigma == 1).sum())
        n_del = int((sigma == -1).sum())
        print(f"sbm snap {i}: {upd_keys.size} events (+{n_ins}, -{n_del})")

    return DynamicOGB(
        name=dataset,
        features=feat,
        labels=raw_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_nodes=N,
        init_edges=init_pairs.astype(np.int32, copy=False),
        snapshots=snapshots,
        label_snapshots=label_snapshots,
    )


def load_dynamic(
    dataset: str,
    root: Optional[str] = None,
    **kwargs,
) -> DynamicOGB:
    """Unified loader: dispatches `dataset` to the right family-specific loader.

    OGB keys (arxiv / products / papers100M) -> `load_ogb_dynamic`.
    SBM keys (sbm-500k)                      -> `load_sbm_dynamic`.

    `root` defaults to a sensible per-family path when None.
    """
    key = dataset.lower()
    if key in _SBM_DEFAULTS:
        # Filter kwargs to those accepted by load_sbm_dynamic.
        sbm_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in {"seed", "feat_dim", "train_frac", "val_frac", "cache_dir"}
        }
        return load_sbm_dynamic(
            dataset=key,
            root=root if root is not None else "data/sbm-500k",
            **sbm_kwargs,
        )
    if key in _OGB_DATASETS:
        ogb_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in {"num_snapshots", "seed", "shuffle_dropped", "drop_mode"}
        }
        return load_ogb_dynamic(
            dataset=key,
            root=root if root is not None else "data/ogb",
            **ogb_kwargs,
        )
    raise ValueError(
        f"Unknown dataset {dataset!r}. Known: "
        f"{list(_OGB_DATASETS) + list(_SBM_DEFAULTS)}"
    )


def load_arxiv_dynamic(
    root: str = "data/ogb",
    num_snapshots: int = 16,
    seed: int = 0,
    shuffle_dropped: bool = True,
    drop_mode: str = "train_node",
) -> DynamicOGB:
    """Backward-compatible wrapper for the arxiv-only loader."""
    return load_ogb_dynamic(
        dataset="arxiv",
        root=root,
        num_snapshots=num_snapshots,
        seed=seed,
        shuffle_dropped=shuffle_dropped,
        drop_mode=drop_mode,
    )


def summarize(ds: DynamicOGB) -> str:
    edges_init = ds.init_edges.shape[0]
    edges_total = edges_init + sum(s.shape[0] for s in ds.snapshots)
    per_snap = [s.shape[0] for s in ds.snapshots]
    return (
        f"{ds.name} dynamic: n={ds.num_nodes}, F={ds.features.shape[1]}, "
        f"edges_init={edges_init}, edges_total={edges_total}, "
        f"num_snapshots={len(ds.snapshots)}, "
        f"per_snap_min/max/mean={min(per_snap)}/{max(per_snap)}/"
        f"{sum(per_snap)/len(per_snap):.1f}"
    )
