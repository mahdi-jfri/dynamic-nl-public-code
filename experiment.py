"""
Experiment harness for comparing linear vs. nonlinear PPR propagation
on node classification. No dynamic machinery — only the nonlinearity
from the paper's Nonlinear PageRank section:

    z_{k+1} = U( alpha * s + (1 - alpha) * W * z_k )

with W = D^{-beta} (A + I) D^{beta-1} and U applied pointwise.

Supported knobs:
  - dataset: cora | citeseer | pubmed | computers | photo | chameleon |
             squirrel | actor | texas | cornell | wisconsin | ogbn-arxiv |
             ogbn-products | ogbn-papers100M
  - feature normalization: none | std | minmax11
  - step_fn: none | sigmoid | tanh | softplus | relu
  - classifier: linear | mlp
  - alpha, beta, ppr_steps, hidden, dropout, lr, wd, epochs, seed

Each run prints a single summary line; a sweep driver (below) aggregates
across seeds and compares linear vs nonlinear within a fixed setting.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import (
    add_self_loops,
    degree,
    remove_self_loops,
    to_undirected,
)

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
                NodeStorage,
                EdgeStorage,
                GlobalStorage,
            ]
        )
    except Exception:
        pass


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


@dataclass
class LoadedData:
    x: torch.Tensor  # [N, F]
    y: torch.Tensor  # [N] long
    edge_index: torch.Tensor  # [2, E], undirected, with self-loops
    num_nodes: int
    num_features: int
    num_classes: int
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    name: str


def _planetoid_split(data, name: str) -> LoadedData:
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    y = data.y.long().view(-1)
    return LoadedData(
        x=data.x.float(),
        y=y,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_features=data.num_features,
        num_classes=int(y.max().item()) + 1,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        name=name,
    )


def _random_per_class_split(
    data, split_idx: int, name: str, train_frac: float = 0.6, val_frac: float = 0.2
) -> LoadedData:
    # For datasets without canonical splits (Amazon Computers/Photo, Coauthor).
    # Stratified random 60/20/20 split per class, seeded by split_idx.
    y = data.y.long().view(-1)
    num_classes = int(y.max().item()) + 1
    g = torch.Generator().manual_seed(split_idx)
    train_list, val_list, test_list = [], [], []
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n = perm.numel()
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        n_tr = min(n_tr, max(0, n - 2))
        n_va = min(n_va, max(0, n - n_tr - 1))
        train_list.append(perm[:n_tr])
        val_list.append(perm[n_tr : n_tr + n_va])
        test_list.append(perm[n_tr + n_va :])
    return LoadedData(
        x=data.x.float(),
        y=y,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_features=data.num_features,
        num_classes=num_classes,
        train_idx=torch.cat(train_list),
        val_idx=torch.cat(val_list),
        test_idx=torch.cat(test_list),
        name=name,
    )


def _geom_gcn_split(data, split_idx: int, name: str) -> LoadedData:
    # WikipediaNetwork(..., geom_gcn_preprocess=True) and Actor provide
    # train/val/test masks of shape [N, 10]; pick a split.
    tr = data.train_mask[:, split_idx].nonzero(as_tuple=False).view(-1)
    va = data.val_mask[:, split_idx].nonzero(as_tuple=False).view(-1)
    te = data.test_mask[:, split_idx].nonzero(as_tuple=False).view(-1)
    y = data.y.long().view(-1)
    return LoadedData(
        x=data.x.float(),
        y=y,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_features=data.num_features,
        num_classes=int(y.max().item()) + 1,
        train_idx=tr,
        val_idx=va,
        test_idx=te,
        name=name,
    )


def load_dataset(name: str, root: str = "data", split_idx: int = 0) -> LoadedData:
    name_l = name.lower()
    _register_pyg_safe_globals()
    if name_l in ("cora", "citeseer", "pubmed"):
        from torch_geometric.datasets import Planetoid

        ds = Planetoid(root=os.path.join(root, "planetoid"), name=name_l.capitalize())
        return _planetoid_split(ds[0], name_l)
    if name_l in ("chameleon", "squirrel"):
        from torch_geometric.datasets import WikipediaNetwork

        ds = WikipediaNetwork(
            root=os.path.join(root, "wikipedia"),
            name=name_l,
            geom_gcn_preprocess=True,
        )
        return _geom_gcn_split(ds[0], split_idx, name_l)
    if name_l in ("chameleon_new", "squirrel_new"):
        # Filtered Chameleon/Squirrel from
        # https://github.com/yandex-research/heterophilous-graphs (Platonov
        # et al., 2023). The original PyG WikipediaNetwork versions have
        # train/test leakage from duplicate nodes; these files remove it.
        import urllib.request

        import numpy as np

        base = name_l.split("_")[0]  # "chameleon" or "squirrel"
        fname = f"{base}_filtered.npz"
        cache_dir = os.path.join(root, "yandex_filtered")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            url = (
                "https://github.com/yandex-research/heterophilous-graphs/"
                f"raw/main/data/{fname}"
            )
            urllib.request.urlretrieve(url, path)
        npz = np.load(path)
        x = torch.from_numpy(npz["node_features"]).float()
        y = torch.from_numpy(npz["node_labels"]).long().view(-1)
        edges = torch.from_numpy(npz["edges"]).long()
        if edges.shape[0] != 2:
            edges = edges.t().contiguous()
        train_masks = torch.from_numpy(npz["train_masks"]).bool()
        val_masks = torch.from_numpy(npz["val_masks"]).bool()
        test_masks = torch.from_numpy(npz["test_masks"]).bool()
        # Masks are [num_splits, N]; pick the requested split.
        s = split_idx % train_masks.shape[0]
        train_idx = train_masks[s].nonzero(as_tuple=False).view(-1)
        val_idx = val_masks[s].nonzero(as_tuple=False).view(-1)
        test_idx = test_masks[s].nonzero(as_tuple=False).view(-1)
        return LoadedData(
            x=x,
            y=y,
            edge_index=edges,
            num_nodes=x.size(0),
            num_features=x.size(1),
            num_classes=int(y.max().item()) + 1,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            name=name_l,
        )
    if name_l == "actor":
        from torch_geometric.datasets import Actor

        ds = Actor(root=os.path.join(root, "actor"))
        return _geom_gcn_split(ds[0], split_idx, name_l)
    if name_l in ("computers", "photo"):
        from torch_geometric.datasets import Amazon

        pretty = {"computers": "Computers", "photo": "Photo"}[name_l]
        ds = Amazon(root=os.path.join(root, "amazon"), name=pretty)
        return _random_per_class_split(ds[0], split_idx, name_l)
    if name_l in (
        "ms-academic",
        "ms_academic",
        "coauthor-cs",
        "cs",
        "coauthor-physics",
        "physics",
    ):
        from torch_geometric.datasets import Coauthor

        pretty = (
            "CS"
            if name_l in ("ms-academic", "ms_academic", "coauthor-cs", "cs")
            else "Physics"
        )
        ds = Coauthor(root=os.path.join(root, "coauthor"), name=pretty)
        return _random_per_class_split(ds[0], split_idx, name_l)
    if name_l in ("cornell", "texas", "wisconsin"):
        from torch_geometric.datasets import WebKB

        ds = WebKB(root=os.path.join(root, "webkb"), name=name_l.capitalize())
        return _geom_gcn_split(ds[0], split_idx, name_l)
    if name_l in (
        "roman-empire",
        "amazon-ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ):
        from torch_geometric.datasets import HeterophilousGraphDataset

        pretty = {
            "roman-empire": "Roman-empire",
            "amazon-ratings": "Amazon-ratings",
            "minesweeper": "Minesweeper",
            "tolokers": "Tolokers",
            "questions": "Questions",
        }[name_l]
        ds = HeterophilousGraphDataset(
            root=os.path.join(root, "hetero"),
            name=pretty,
        )
        return _geom_gcn_split(ds[0], split_idx, name_l)
    if name_l in ("ogbn-arxiv", "ogbn-products", "ogbn-papers100m", "products", "papers100m"):
        import builtins

        from ogb.nodeproppred import PygNodePropPredDataset

        ogb_name = {
            "ogbn-arxiv": "ogbn-arxiv",
            "ogbn-products": "ogbn-products",
            "products": "ogbn-products",
            "ogbn-papers100m": "ogbn-papers100M",
            "papers100m": "ogbn-papers100M",
        }[name_l]
        # OGB prompts via input() when its local cache version is stale; auto-confirm
        # so non-interactive shard runs don't EOF.
        _real_input = builtins.input
        builtins.input = lambda *a, **kw: "y"
        # OGB's cached processed files are pickled with numpy scalars; torch>=2.6
        # defaults weights_only=True and refuses to load them.
        _real_torch_load = torch.load
        torch.load = lambda *a, **kw: _real_torch_load(*a, **{**kw, "weights_only": False})
        try:
            ds = PygNodePropPredDataset(name=ogb_name, root=os.path.join(root, "ogb"))
        finally:
            builtins.input = _real_input
            torch.load = _real_torch_load
        split = ds.get_idx_split()
        data = ds[0]
        y = data.y.long().view(-1)
        return LoadedData(
            x=data.x.float(),
            y=y,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_features=data.num_features,
            num_classes=int(y.max().item()) + 1,
            train_idx=split["train"],
            val_idx=split["valid"],
            test_idx=split["test"],
            name=ogb_name,
        )
    raise ValueError(f"Unknown dataset: {name}")


# ----------------------------------------------------------------------
# Feature normalization
# ----------------------------------------------------------------------


def normalize_features(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "std":
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, keepdim=True).clamp_min(1e-12)
        return (x - mu) / sd
    if mode == "minmax11":
        mn = x.min(dim=0, keepdim=True).values
        mx = x.max(dim=0, keepdim=True).values
        return 2.0 * (x - mn) / (mx - mn).clamp_min(1e-12) - 1.0
    raise ValueError(f"Unknown normalization mode: {mode}")


# ----------------------------------------------------------------------
# Adjacency + propagation
# ----------------------------------------------------------------------


def build_norm_adj(
    edge_index: torch.Tensor, num_nodes: int, device: torch.device, beta: float
) -> torch.Tensor:
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_left = deg.pow(-beta)
    deg_right = deg.pow(beta - 1.0)
    deg_left[torch.isinf(deg_left)] = 0.0
    deg_right[torch.isinf(deg_right)] = 0.0
    values = deg_left[row] * deg_right[col]
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()
    return adj


def _step_fn(name: str):
    if name == "none":
        return lambda t: t
    if name == "sigmoid":
        return torch.sigmoid
    if name == "tanh":
        return torch.tanh
    if name == "softplus":
        return F.softplus
    if name == "relu":
        return F.relu
    if name == "clamp":
        # 1-Lipschitz, identity on [-1,1], saturates outside.
        return lambda t: torch.clamp(t, min=-1.0, max=1.0)
    if name == "clamp01":
        return lambda t: torch.clamp(t, min=0.0, max=1.0)
    if name == "leaky":
        # 1-Lipschitz piecewise-linear with slope 0.5 on negatives.
        return lambda t: F.leaky_relu(t, negative_slope=0.5)
    if name.startswith("soft"):
        # soft-thresholding f(x) = sign(x) * max(|x| - t, 0), 1-Lipschitz.
        try:
            thresh = float(name[len("soft") :]) if name != "soft" else 0.1
        except ValueError:
            thresh = 0.1
        return lambda z, t=thresh: torch.sign(z) * F.relu(torch.abs(z) - t)
    if name.startswith("htanh"):
        # Hard-tanh = clamp to [-w, w]; the param is width.
        try:
            w = float(name[len("htanh") :]) if name != "htanh" else 1.0
        except ValueError:
            w = 1.0
        return lambda z, w=w: torch.clamp(z, min=-w, max=w)
    if name.startswith("stanh"):
        # Scaled tanh: f(x) = tanh(c*x)/c. Derivative sech^2(c*x), bounded
        # by 1, so this is 1-Lipschitz for any c>0. Interpolates between
        # identity (c->0) and sign(x)/c (c large).
        try:
            c = float(name[len("stanh") :]) if name != "stanh" else 1.0
        except ValueError:
            c = 1.0
        return lambda z, c=c: torch.tanh(c * z) / c
    if name.startswith("shtanh"):
        # Shifted tanh: f(x) = tanh(x - b) — 1-Lipschitz, encodes a bias.
        try:
            b = float(name[len("shtanh") :]) if name != "shtanh" else 0.0
        except ValueError:
            b = 0.0
        return lambda z, b=b: torch.tanh(z - b)
    if name.startswith("scl"):
        # Symmetric scaled clamp: clamp(x, -c, c). 1-Lipschitz.
        try:
            c = float(name[len("scl") :]) if name != "scl" else 1.0
        except ValueError:
            c = 1.0
        return lambda z, c=c: torch.clamp(z, min=-c, max=c)
    if name.startswith("abs"):
        # |x| capped at 1 is not 1-Lipschitz (kink), but
        # |x|_shrink(x) = sign(x)*max(|x|-t,0) is covered by "soft".
        raise ValueError("use soft<t> for soft-thresholding")
    if name == "halfplus":
        # f(x) = max(x, 0) + 0.5 * min(x, 0). Slope 1 on positives, 0.5
        # on negatives, 1-Lipschitz. Same as leaky_relu(0.5).
        return lambda z: F.leaky_relu(z, negative_slope=0.5)
    if name.startswith("ltanh"):
        lambda_, c = map(float, name.split("_")[1:])
        return lambda z, lambda_=lambda_, c=c: (1 - lambda_) * z + lambda_ * torch.tanh(c * z) / c
    raise ValueError(f"Unknown step_fn: {name}")


@torch.no_grad()
def ppr_diffusion(
    x: torch.Tensor,
    adj: torch.Tensor,
    alpha: float,
    steps: int,
    step_fn: str,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Fixed-point iteration: z_{k+1} = U(alpha * x + (1-alpha) * A * z_k).

    Starts from z_0 = 0 (the canonical start used in the paper's
    Algorithm 1), so the first iterate is U(alpha * x).
    """
    f = _step_fn(step_fn)
    z = torch.zeros_like(x)
    y = alpha * x
    for k in range(steps):
        y = alpha * x + (1.0 - alpha) * torch.sparse.mm(adj, z)
        z_next = f(y)
        if torch.norm(z_next - z) <= epsilon:
            z = z_next
            break
        z = z_next
    return torch.cat([y, z], dim=1)


# ----------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------


class LinearClassifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.lin(x)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden)
        self.lin2 = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


@dataclass
class RunConfig:
    dataset: str = "cora"
    normalize: str = "row"
    feat_scale: float = 1.0
    step_fn: str = "none"
    classifier: str = "linear"
    alpha: float = 0.1
    beta: float = 1.0
    ppr_steps: int = 50
    hidden: int = 256
    dropout: float = 0.5
    lr: float = 0.01
    wd: float = 5e-4
    epochs: int = 300
    patience: int = 100
    seed: int = 0
    split_idx: int = 0
    device: str = "cuda"


@dataclass
class RunResult:
    best_val: float
    best_test: float
    best_epoch: int
    ppr_time: float
    train_time: float


from collections import OrderedDict

_PPR_CACHE: "OrderedDict" = OrderedDict()
_PPR_CACHE_MAX = int(os.environ.get("PPR_CACHE_MAX", "2"))
_ADJ_CACHE: dict = {}


def _data_key(data: LoadedData) -> tuple:
    # Stable key independent of Python id() reuse.
    return (
        data.name,
        int(data.num_nodes),
        int(data.num_features),
        int(data.num_classes),
        int(data.train_idx.numel()),
        int(data.val_idx.numel()),
        int(data.test_idx.numel()),
    )


def _get_adj(data: LoadedData, beta: float, device: torch.device) -> torch.Tensor:
    key = (_data_key(data), float(beta), device.type, device.index)
    if key in _ADJ_CACHE:
        return _ADJ_CACHE[key]
    edge_index = None
    retries = 20
    for i in range(retries):
        try:
            edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
            edge_index = edge_index.to(device)
            break
        except Exception as e:
            print(
                f"Warning: failed to process edge_index on attempt {i+1}/20: {e}",
                file=sys.stderr,
                flush=True,
            )
            if i == retries - 1:
                raise
    adj = build_norm_adj(edge_index, data.num_nodes, device, beta=beta)
    _ADJ_CACHE[key] = adj
    return adj


def _get_ppr(
    data: LoadedData,
    normalize: str,
    feat_scale: float,
    alpha: float,
    beta: float,
    steps: int,
    step_fn: str,
    device: torch.device,
) -> torch.Tensor:
    key = (
        _data_key(data),
        normalize,
        float(feat_scale),
        float(alpha),
        float(beta),
        int(steps),
        step_fn,
        device.type,
        device.index,
    )
    if key in _PPR_CACHE:
        _PPR_CACHE.move_to_end(key)
        return _PPR_CACHE[key]
    x = normalize_features(data.x, normalize).to(device) * float(feat_scale)
    adj = _get_adj(data, beta, device)
    z = ppr_diffusion(x, adj, alpha=alpha, steps=steps, step_fn=step_fn)
    _PPR_CACHE[key] = z
    while len(_PPR_CACHE) > _PPR_CACHE_MAX:
        _PPR_CACHE.popitem(last=False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return z


def clear_caches():
    _PPR_CACHE.clear()
    _ADJ_CACHE.clear()


def run_once(cfg: RunConfig, data: LoadedData) -> RunResult:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    y = data.y.to(device)

    t0 = time.time()
    z = _get_ppr(
        data,
        cfg.normalize,
        cfg.feat_scale,
        cfg.alpha,
        cfg.beta,
        cfg.ppr_steps,
        cfg.step_fn,
        device,
    )
    ppr_time = time.time() - t0

    # Seed AFTER PPR so classifier init depends only on cfg.seed.
    set_seed(cfg.seed)

    if cfg.classifier == "linear":
        model = LinearClassifier(z.size(1), data.num_classes).to(device)
    elif cfg.classifier == "mlp":
        model = MLP(z.size(1), cfg.hidden, data.num_classes, cfg.dropout).to(device)
    else:
        raise ValueError(cfg.classifier)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    test_idx = data.test_idx.to(device)

    best_epoch = 0
    best_val = -1.0
    best_test = 0.0

    t1 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optim.zero_grad()
        out = model(z)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            out = model(z)
            pred = out.argmax(dim=-1)
            val_acc = float((pred[val_idx] == y[val_idx]).float().mean().item())
            test_acc = float((pred[test_idx] == y[test_idx]).float().mean().item())
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch

        if cfg.patience > 0 and epoch - best_epoch >= cfg.patience:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t1

    return RunResult(
        best_val=best_val,
        best_test=best_test,
        best_epoch=best_epoch,
        ppr_time=ppr_time,
        train_time=train_time,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora")
    p.add_argument("--normalize", default="std")
    p.add_argument("--step-fn", default="none")
    p.add_argument("--classifier", default="linear", choices=["linear", "mlp"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ppr-steps", type=int, default=50)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--split-idx", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    args = build_argparser().parse_args()
    data = load_dataset(args.dataset, split_idx=args.split_idx)
    cfg = RunConfig(
        dataset=args.dataset,
        normalize=args.normalize,
        step_fn=args.step_fn,
        classifier=args.classifier,
        alpha=args.alpha,
        beta=args.beta,
        ppr_steps=args.ppr_steps,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        split_idx=args.split_idx,
        device=args.device,
    )
    accs = []
    for s in args.seeds:
        cfg.seed = s
        r = run_once(cfg, data)
        accs.append(r.best_test)
        print(
            f"[{args.dataset} norm={args.normalize} step={args.step_fn} "
            f"cls={args.classifier} a={args.alpha} b={args.beta} "
            f"steps={args.ppr_steps} seed={s}] "
            f"val={r.best_val:.4f} test={r.best_test:.4f} "
            f"ppr={r.ppr_time:.2f}s train={r.train_time:.2f}s",
            flush=True,
        )
    t = torch.tensor(accs)
    print(
        f"SUMMARY {args.dataset} norm={args.normalize} step={args.step_fn} "
        f"cls={args.classifier} a={args.alpha} b={args.beta} "
        f"steps={args.ppr_steps} mean={t.mean():.4f} std={t.std():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
