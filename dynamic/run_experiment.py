"""Driver for the dynamic graph experiments.

Three modes, picked with --mode:
  dynamic      - maintain (z, y, r) across the stream via Algorithm 1.
  from_scratch - advance the graph only, then reset state + cleanup each snapshot.
  both         - run dynamic and from_scratch side-by-side, verify z agrees,
                 and print per-snapshot timing for both.

Per snapshot we (a) update propagation, (b) retrain the MLP head on the new z,
    and log best-val / test-at-best-val.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from dynamic import NonlinearPPR, resolve_step_fn, resolve_threshold_mode
from dynamic.classifier import train_mlp
from dynamic.dataset import DynamicOGB, load_dynamic, summarize


@dataclass
class SnapResult:
    snapshot: int  # 0 = init, 1..num_snap = per-snapshot
    mode: str  # "dynamic" | "from_scratch" | "instantgnn"
    prop_time: float  # seconds spent in propagation (cleanup / snapshot_op)
    edge_events: int  # count of edge events applied this step
    val_acc: float
    test_acc: float
    train_time: float
    residual_l1: float


def _make_alg(
    ds: DynamicOGB,
    alpha: float,
    beta: float,
    step_fn: str,
    step_param: float,
    K_override: float,
    threshold_mode: str = "degree",
    feat_scale: float = 1.0,
) -> NonlinearPPR:
    sf_enum, K_default = resolve_step_fn(step_fn)
    K = K_override if K_override > 0 else K_default
    tm_enum = resolve_threshold_mode(threshold_mode)
    N = ds.num_nodes
    F = ds.features.shape[1]
    alg = NonlinearPPR(
        n=N,
        F=F,
        alpha=alpha,
        beta=beta,
        K=K,
        step_fn=int(sf_enum),
        step_param=step_param,
        threshold_mode=int(tm_enum),
    )
    feat = ds.features
    if feat_scale != 1.0:
        feat = feat * float(feat_scale)
    # [F, N] column-major input to the C++ side.
    alg.set_features(np.ascontiguousarray(feat.T))
    # C++ owns the feature copy now; release the [N, F] array on the dataset
    # so we don't hold ~114 GB on papers100M for nothing.
    del feat
    ds.features = None
    return alg


def _alg_z_nf(alg: NonlinearPPR, step_fn: str) -> np.ndarray:
    z = alg.get_z()  # [F, N]
    z_nf = np.ascontiguousarray(z.T, dtype=np.float32)  # [N, F]
    if step_fn != "none":
        y = alg.get_y()  # [F, N]
        y_nf = np.ascontiguousarray(y.T, dtype=np.float32)  # [N, F]
        z_nf = np.concatenate([z_nf, y_nf], axis=1)  # [N, 2F]
    return z_nf


def _labels_for_snapshot(ds: DynamicOGB, snapshot: int) -> np.ndarray:
    """Return labels for snapshot 0=init, 1..M=post-update labels."""
    if snapshot == 0 or ds.label_snapshots is None:
        return ds.labels
    return ds.label_snapshots[snapshot - 1]


def _num_classes(ds: DynamicOGB) -> int:
    if ds.label_snapshots is None:
        return int(ds.labels.max() + 1)
    return int(
        max(ds.labels.max(), max(labels.max() for labels in ds.label_snapshots)) + 1
    )


def _train_from_z_nf(
    ds: DynamicOGB,
    z_nf: np.ndarray,
    *,
    snapshot: int,
    num_classes: int,
    seed: int,
    mlp_hidden: int,
    mlp_num_layers: int,
    mlp_lr: float,
    mlp_wd: float,
    mlp_epochs: int,
    mlp_dropout: float,
    device: str,
    batch_size: int = 0,
):
    return train_mlp(
        z_nf,
        _labels_for_snapshot(ds, snapshot=snapshot),
        ds.train_idx,
        ds.val_idx,
        ds.test_idx,
        num_classes=num_classes,
        hidden=mlp_hidden,
        num_layers=mlp_num_layers,
        dropout=mlp_dropout,
        lr=mlp_lr,
        wd=mlp_wd,
        epochs=mlp_epochs,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )


def _run_and_eval(
    ds: DynamicOGB,
    alg: NonlinearPPR,
    *,
    snapshot: int,
    num_classes: int,
    seed: int,
    mlp_hidden: int,
    mlp_num_layers: int,
    mlp_lr: float,
    mlp_wd: float,
    mlp_epochs: int,
    mlp_dropout: float,
    device: str,
    step_fn: str,
    batch_size: int = 0,
):
    z_nf = _alg_z_nf(alg, step_fn)
    res = _train_from_z_nf(
        ds,
        z_nf,
        snapshot=snapshot,
        num_classes=num_classes,
        seed=seed,
        mlp_hidden=mlp_hidden,
        mlp_num_layers=mlp_num_layers,
        mlp_lr=mlp_lr,
        mlp_wd=mlp_wd,
        mlp_epochs=mlp_epochs,
        mlp_dropout=mlp_dropout,
        device=device,
        batch_size=batch_size,
    )
    return res, float(alg.total_residual_l1())


def _snap_path(prop_dir: str, mode: str, snapshot: int) -> str:
    return os.path.join(prop_dir, f"{mode}_snap_{snapshot:03d}.npz")


def _meta_path(prop_dir: str, mode: str) -> str:
    return os.path.join(prop_dir, f"{mode}_meta.json")


def _splits_path(prop_dir: str) -> str:
    # Shared across modes: labels and train/val/test indices only depend on
    # the dataset/seed/drop_mode, not on the propagation mode.
    return os.path.join(prop_dir, "splits.npz")


def _save_splits(prop_dir: str, ds: DynamicOGB) -> None:
    data = dict(
        labels=ds.labels,
        train_idx=ds.train_idx,
        val_idx=ds.val_idx,
        test_idx=ds.test_idx,
        num_nodes=np.int64(ds.num_nodes),
        num_classes=np.int64(_num_classes(ds)),
    )
    if ds.label_snapshots is not None:
        data["label_snapshots"] = np.stack(ds.label_snapshots, axis=0)
    np.savez_compressed(_splits_path(prop_dir), **data)


def _load_splits(prop_dir: str):
    """Returns (labels, label_snapshots, train_idx, val_idx, test_idx, num_classes)."""
    d = np.load(_splits_path(prop_dir))
    label_snapshots = d["label_snapshots"] if "label_snapshots" in d else None
    return (
        d["labels"],
        label_snapshots,
        d["train_idx"],
        d["val_idx"],
        d["test_idx"],
        int(d["num_classes"]),
    )


def run_mode(
    ds: DynamicOGB,
    *,
    mode: str,
    alpha: float,
    beta: float,
    eps: float,
    step_fn: str,
    step_param: float,
    K_override: float,
    mlp_hidden: int,
    mlp_num_layers: int,
    mlp_lr: float,
    mlp_wd: float,
    mlp_epochs: int,
    mlp_dropout: float,
    seed: int,
    device: str,
    threshold_mode: str = "degree",
    feat_scale: float = 1.0,
    batch_size: int = 0,
) -> List[SnapResult]:
    assert mode in ("dynamic", "dynamic_batched", "from_scratch", "from_scratch_edge")
    num_classes = _num_classes(ds)
    results: List[SnapResult] = []

    alg = _make_alg(
        ds,
        alpha,
        beta,
        step_fn,
        step_param,
        K_override,
        threshold_mode=threshold_mode,
        feat_scale=feat_scale,
    )

    # Initial operation is common to both modes: build the initial graph and
    # run a full cleanup from the canonical init state.
    t0 = time.time()
    alg.initial_operation(ds.init_edges, eps)
    t_prop = time.time() - t0
    res, rl1 = _run_and_eval(
        ds,
        alg,
        snapshot=0,
        num_classes=num_classes,
        seed=seed,
        mlp_hidden=mlp_hidden,
        mlp_num_layers=mlp_num_layers,
        mlp_lr=mlp_lr,
        mlp_wd=mlp_wd,
        mlp_epochs=mlp_epochs,
        mlp_dropout=mlp_dropout,
        device=device,
        step_fn=step_fn,
        batch_size=batch_size,
    )
    results.append(
        SnapResult(
            snapshot=0,
            mode=mode,
            prop_time=t_prop,
            edge_events=int(ds.init_edges.shape[0]),
            val_acc=res.best_val,
            test_acc=res.best_test,
            train_time=res.train_time,
            residual_l1=rl1,
        )
    )
    print(
        f"[{mode} init] prop={t_prop:.3f}s val={res.best_val:.4f} "
        f"test={res.best_test:.4f} r_l1={rl1:.3e} train_time={res.train_time:.1f}s",
        flush=True,
    )

    for i, ev in enumerate(ds.snapshots, start=1):
        t0 = time.time()
        if mode == "dynamic":
            alg.snapshot_operation(ev, eps)
        elif mode == "dynamic_batched":
            alg.snapshot_operation_batched(ev, eps)
        elif mode == "from_scratch_edge":
            # Per-edge from-scratch: for each event, apply that single edge,
            # reset state, and run a full cleanup. One cleanup per edge.
            ev2 = np.ascontiguousarray(ev, dtype=np.int32)
            for k in range(ev2.shape[0]):
                alg.apply_edge_events(ev2[k : k + 1])
                alg.reset_state()
                alg.cleanup(eps)
        else:
            # From scratch: advance graph, reset state, cleanup from canonical.
            alg.apply_edge_events(ev)
            alg.reset_state()
            alg.cleanup(eps)
        t_prop = time.time() - t0
        res, rl1 = _run_and_eval(
            ds,
            alg,
            snapshot=i,
            num_classes=num_classes,
            seed=seed,
            mlp_hidden=mlp_hidden,
            mlp_lr=mlp_lr,
            mlp_wd=mlp_wd,
            mlp_epochs=mlp_epochs,
            mlp_dropout=mlp_dropout,
            device=device,
            mlp_num_layers=mlp_num_layers,
            step_fn=step_fn,
            batch_size=batch_size,
        )
        results.append(
            SnapResult(
                snapshot=i,
                mode=mode,
                prop_time=t_prop,
                edge_events=int(ev.shape[0]),
                val_acc=res.best_val,
                test_acc=res.best_test,
                train_time=res.train_time,
                residual_l1=rl1,
            )
        )
        print(
            f"[{mode} snap {i:02d}] events={ev.shape[0]} "
            f"prop={t_prop:.3f}s val={res.best_val:.4f} "
            f"test={res.best_test:.4f} r_l1={rl1:.3e} train_time={res.train_time:.1f}s",
            flush=True,
        )
    return results


def run_propagate(
    ds: DynamicOGB,
    *,
    mode: str,
    alpha: float,
    beta: float,
    eps: float,
    step_fn: str,
    step_param: float,
    K_override: float,
    threshold_mode: str,
    feat_scale: float,
    prop_dir: str,
) -> List[dict]:
    """Run propagation only and save (z, y) per snapshot to disk."""
    assert mode in ("dynamic", "dynamic_batched", "from_scratch", "from_scratch_edge")
    os.makedirs(prop_dir, exist_ok=True)
    # Cache labels/splits so the train stage doesn't need to load the dataset.
    _save_splits(prop_dir, ds)
    alg = _make_alg(
        ds,
        alpha,
        beta,
        step_fn,
        step_param,
        K_override,
        threshold_mode=threshold_mode,
        feat_scale=feat_scale,
    )

    meta: List[dict] = []

    t0 = time.time()
    alg.initial_operation(ds.init_edges, eps)
    t_prop = time.time() - t0
    rl1 = float(alg.total_residual_l1())
    np.savez_compressed(_snap_path(prop_dir, mode, 0), z_nf=_alg_z_nf(alg, step_fn))
    meta.append(
        dict(
            snapshot=0,
            mode=mode,
            prop_time=t_prop,
            edge_events=int(ds.init_edges.shape[0]),
            residual_l1=rl1,
        )
    )
    print(
        f"[{mode} init] prop={t_prop:.3f}s r_l1={rl1:.3e} " f"saved snap 0", flush=True
    )

    for i, ev in enumerate(ds.snapshots, start=1):
        t0 = time.time()
        if mode == "dynamic":
            alg.snapshot_operation(ev, eps)
        elif mode == "dynamic_batched":
            alg.snapshot_operation_batched(ev, eps)
        elif mode == "from_scratch_edge":
            ev2 = np.ascontiguousarray(ev, dtype=np.int32)
            for k in range(ev2.shape[0]):
                alg.apply_edge_events(ev2[k : k + 1])
                alg.reset_state()
                alg.cleanup(eps)
        else:
            alg.apply_edge_events(ev)
            alg.reset_state()
            alg.cleanup(eps)
        t_prop = time.time() - t0
        rl1 = float(alg.total_residual_l1())
        np.savez(_snap_path(prop_dir, mode, i), z_nf=_alg_z_nf(alg, step_fn))
        meta.append(
            dict(
                snapshot=i,
                mode=mode,
                prop_time=t_prop,
                edge_events=int(ev.shape[0]),
                residual_l1=rl1,
            )
        )
        print(
            f"[{mode} snap {i:02d}] events={ev.shape[0]} "
            f"prop={t_prop:.3f}s r_l1={rl1:.3e} saved snap {i}",
            flush=True,
        )

    with open(_meta_path(prop_dir, mode), "w") as f:
        json.dump(
            dict(
                mode=mode,
                alpha=alpha,
                beta=beta,
                eps=eps,
                step_fn=step_fn,
                step_param=step_param,
                num_classes=_num_classes(ds),
                num_snapshots=len(ds.snapshots),
                snapshots=meta,
            ),
            f,
            indent=2,
        )
    print(f"wrote propagation state to {prop_dir} ({mode})", flush=True)
    return meta


def run_train_one(
    *,
    mode: str,
    snapshot: int,
    prop_dir: str,
    mlp_hidden: int,
    mlp_num_layers: int,
    mlp_lr: float,
    mlp_wd: float,
    mlp_epochs: int,
    mlp_dropout: float,
    seed: int,
    device: str,
    batch_size: int = 0,
) -> SnapResult:
    """Load saved propagation state + cached splits for one snapshot and
    train the MLP. Does NOT load the OGB dataset."""
    if not os.path.exists(_splits_path(prop_dir)):
        raise SystemExit(
            f"missing {_splits_path(prop_dir)} - run --stage propagate "
            "first (it now caches labels/splits)."
        )
    labels, label_snapshots, train_idx, val_idx, test_idx, num_classes = _load_splits(
        prop_dir
    )
    if snapshot > 0 and label_snapshots is not None:
        labels = label_snapshots[snapshot - 1]
    path = _snap_path(prop_dir, mode, snapshot)
    z_nf = np.load(path)["z_nf"]
    meta_p = _meta_path(prop_dir, mode)
    snap_meta = dict(prop_time=float("nan"), edge_events=0, residual_l1=float("nan"))
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            m = json.load(f)
        for s in m.get("snapshots", []):
            if int(s["snapshot"]) == int(snapshot):
                snap_meta = s
                break
    res = train_mlp(
        z_nf,
        labels,
        train_idx,
        val_idx,
        test_idx,
        num_classes=num_classes,
        hidden=mlp_hidden,
        num_layers=mlp_num_layers,
        dropout=mlp_dropout,
        lr=mlp_lr,
        wd=mlp_wd,
        epochs=mlp_epochs,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    out = SnapResult(
        snapshot=snapshot,
        mode=mode,
        prop_time=float(snap_meta.get("prop_time", float("nan"))),
        edge_events=int(snap_meta.get("edge_events", 0)),
        val_acc=res.best_val,
        test_acc=res.best_test,
        train_time=res.train_time,
        residual_l1=float(snap_meta.get("residual_l1", float("nan"))),
    )
    print(
        f"[{mode} snap {snapshot:02d} train] val={res.best_val:.4f} "
        f"test={res.best_test:.4f} train_time={res.train_time:.1f}s",
        flush=True,
    )
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=[
            "dynamic",
            "dynamic_batched",
            "from_scratch",
            "both",
            "from_scratch_edge",
        ],
        default="both",
    )
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--step-fn", default="tanh")
    p.add_argument("--step-param", type=float, default=1.0)
    p.add_argument(
        "--K",
        type=float,
        default=0.0,
        help="Override Lipschitz constant K. 0 = use the default "
        "tied to --step-fn (tanh/relu/clamp=1, sigmoid=0.25).",
    )
    p.add_argument(
        "--dataset",
        choices=["arxiv", "products", "papers100M", "sbm-500k"],
        default="arxiv",
        help="Dataset to load. OGB defaults: "
        "arxiv=16 snaps, products=15, papers100M=20. "
        "sbm-500k loads the synthetic SBM (10 snapshots) from "
        "data/sbm-500k/.",
    )
    p.add_argument(
        "--num-snapshots",
        type=int,
        default=0,
        help="Override num snapshots (OGB only). 0 = use the "
        "per-dataset default (arxiv=16, "
        "products=15, papers100M=20).",
    )
    p.add_argument(
        "--drop-mode",
        choices=["train_node", "random"],
        default="train_node",
        help="OGB-only: how to pick edges that get streamed in.",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Per-family default if unset: "
        "OGB -> 'data/ogb', SBM -> 'data/sbm-500k'.",
    )
    p.add_argument(
        "--feat-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to features after StandardScaler.",
    )
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument(
        "--mlp-num-layers",
        type=int,
        default=2,
        help="Total number of linear layers in the MLP head "
        "(>=2). Hidden layers use BatchNorm1d + ReLU + "
        "dropout.",
    )
    p.add_argument("--mlp-dropout", type=float, default=0.3)
    p.add_argument("--mlp-lr", type=float, default=1e-4)
    p.add_argument("--mlp-wd", type=float, default=0.0)
    p.add_argument("--mlp-epochs", type=int, default=300)
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Minibatch size for MLP training. 0 = full-batch " "(default).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "-t",
        "--threshold-mode",
        choices=["degree", "rowsum"],
        default="degree",
        help="Push threshold. 'degree': "
        "(1-K(1-alpha))*eps*d^(1-beta) per node (Alg-1 proof). "
        "'rowsum': rowsum_{pos,neg}[d]*eps per dim "
        "(looser for "
        "StandardScaler'd features).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to dump per-snapshot results as JSONL.",
    )
    p.add_argument(
        "--stage",
        choices=["full", "propagate", "train"],
        default="full",
        help="full (default) = original behavior. "
        "propagate = run propagation only and save (z[,y]) "
        "per snapshot to --prop-dir. train = load saved "
        "snapshot --snapshot from --prop-dir and run MLP.",
    )
    p.add_argument(
        "--prop-dir",
        default=None,
        help="Directory for saved propagation state (used by "
        "--stage propagate / --stage train).",
    )
    p.add_argument(
        "--snapshot",
        type=int,
        default=-1,
        help="Snapshot index to train when --stage=train "
        "(0 = init, 1.. = post-snapshot).",
    )
    return p


def main():
    args = build_argparser().parse_args()
    print(args)

    if args.stage in ("propagate", "train") and args.mode == "both":
        raise SystemExit(
            "--stage propagate/train requires a single --mode " "(not 'both')."
        )
    if args.stage in ("propagate", "train") and not args.prop_dir:
        raise SystemExit("--stage propagate/train requires --prop-dir.")

    if args.stage == "train":
        # Train stage reads only the cached splits + saved z_nf from --prop-dir.
        # It does NOT load the OGB dataset.
        if args.snapshot < 0:
            raise SystemExit("--stage train requires --snapshot >= 0.")
        r = run_train_one(
            mode=args.mode,
            snapshot=args.snapshot,
            prop_dir=args.prop_dir,
            mlp_hidden=args.mlp_hidden,
            mlp_num_layers=args.mlp_num_layers,
            mlp_lr=args.mlp_lr,
            mlp_wd=args.mlp_wd,
            mlp_epochs=args.mlp_epochs,
            mlp_dropout=args.mlp_dropout,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
        )
        if args.out:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True
            )
            with open(args.out, "a") as f:
                f.write(
                    json.dumps(
                        {
                            **asdict(r),
                            "alpha": args.alpha,
                            "beta": args.beta,
                            "eps": args.eps,
                            "step_fn": args.step_fn,
                            "step_param": args.step_param,
                            "seed": args.seed,
                        }
                    )
                    + "\n"
                )
            print(f"appended to {args.out}", flush=True)
        os.remove(_snap_path(args.prop_dir, args.mode, args.snapshot))
        return

    print(f"loading {args.dataset} (dynamic split)...", flush=True)
    ds = load_dynamic(
        dataset=args.dataset,
        root=args.dataset_root,
        num_snapshots=args.num_snapshots if args.num_snapshots > 0 else None,
        seed=args.seed,
        drop_mode=args.drop_mode,
    )
    print(summarize(ds), flush=True)

    if args.stage == "propagate":
        run_propagate(
            ds,
            mode=args.mode,
            alpha=args.alpha,
            beta=args.beta,
            eps=args.eps,
            step_fn=args.step_fn,
            step_param=args.step_param,
            K_override=args.K,
            threshold_mode=args.threshold_mode,
            feat_scale=args.feat_scale,
            prop_dir=args.prop_dir,
        )
        return

    all_results: List[SnapResult] = []
    modes = ["dynamic", "from_scratch"] if args.mode == "both" else [args.mode]
    for m in modes:
        print(
            f"\n=== mode={m} alpha={args.alpha} beta={args.beta} "
            f"eps={args.eps} step_fn={args.step_fn} ===",
            flush=True,
        )
        rs = run_mode(
            ds,
            mode=m,
            alpha=args.alpha,
            beta=args.beta,
            eps=args.eps,
            step_fn=args.step_fn,
            step_param=args.step_param,
            K_override=args.K,
            mlp_hidden=args.mlp_hidden,
            mlp_num_layers=args.mlp_num_layers,
            mlp_lr=args.mlp_lr,
            mlp_wd=args.mlp_wd,
            mlp_epochs=args.mlp_epochs,
            mlp_dropout=args.mlp_dropout,
            seed=args.seed,
            device=args.device,
            threshold_mode=args.threshold_mode,
            feat_scale=args.feat_scale,
            batch_size=args.batch_size,
        )
        all_results.extend(rs)

        total_prop = sum(r.prop_time for r in rs)
        print(
            f"[{m} TOTAL] prop_time={total_prop:.2f}s over {len(rs)} steps", flush=True
        )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for r in all_results:
                f.write(
                    json.dumps(
                        {
                            **asdict(r),
                            "alpha": args.alpha,
                            "beta": args.beta,
                            "eps": args.eps,
                            "step_fn": args.step_fn,
                            "step_param": args.step_param,
                            "seed": args.seed,
                        }
                    )
                    + "\n"
                )
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
