"""
Best-vs-best comparison of linear vs. nonlinear static Nonlinear PageRank
feature propagation. For each step function U (including the linear one)
we sweep the same hyperparameter grid, select the config with the highest
mean validation accuracy across (split x seed), and report its held-out
test accuracy, its mean/SE per-split, and a paired t-test of the selected
nonlinear variant against the selected linear variant across splits.

Modes
-----
1) Single-process (default): runs the full grid in one process, writes
   a per-run TSV log, then does model-selection + paired stats in place.
       python bvb.py --dataset cornell --log /tmp/bvb.log

2) Shard mode: one process runs only the units whose global index
   satisfies `i % num_shards == shard_id`. Writes a per-shard TSV log
   (no analysis). Use this to parallelize over several local processes.
       python bvb.py --dataset cornell --log /tmp/bvb.log \\
            --shard 0 --num-shards 4

3) Merge mode: reads a set of shard TSV logs and performs model
   selection + paired stats, writing the summary to --log.
       python bvb.py --merge /tmp/bvb.log.shard0 /tmp/bvb.log.shard1 ...\\
            --log /tmp/bvb.log --dataset cornell

The `scripts/run_static_bvb.sh` launcher wraps (2) and (3).
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import List, Tuple

import torch

from experiment import RunConfig, load_dataset, run_once

# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------


def _paired_stats(diffs):
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(diffs) / n
    if n < 2:
        return mean, 0.0, float("nan")
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    return mean, se, t


# ----------------------------------------------------------------------
# Grid enumeration
# ----------------------------------------------------------------------


def _build_grid(args):
    """Return (hparam_product, cls_product, splits)."""
    ds = args.dataset.lower()
    geom = ds in ("chameleon", "squirrel", "actor", "cornell", "texas", "wisconsin")
    if args.splits is not None:
        splits = args.splits
    elif geom:
        splits = list(range(10))
    else:
        splits = [0]

    hparam_product = list(
        itertools.product(
            args.normalizations,
            args.feat_scales,
            args.alphas,
            args.betas,
            args.steps_list,
        )
    )

    cls_product = []
    for cls in args.classifiers:
        if cls == "linear":
            for lr in args.lrs:
                for wd in args.wds:
                    cls_product.append((cls, 0, 0.0, lr, wd))
        else:
            for hidden in args.hiddens:
                for dropout in args.dropouts:
                    for lr in args.lrs:
                        for wd in args.wds:
                            cls_product.append((cls, hidden, dropout, lr, wd))
    return hparam_product, cls_product, splits


def _iter_units(args, hparam_product, cls_product, splits):
    """Flat iterator over (hparam, cls_cfg, fn, split). Seeds stay as an
    inner loop inside each unit so the PPR cache warms up once per unit."""
    for hp in hparam_product:
        for fn in args.step_fns:
            for cls_cfg in cls_product:
                for s_idx in splits:
                    yield (hp, cls_cfg, fn, s_idx)


# ----------------------------------------------------------------------
# Core run loop
# ----------------------------------------------------------------------


_TSV_HEADER = (
    "fn\tnorm\tscale\talpha\tbeta\tsteps\tcls\thidden\tdropout"
    "\tlr\twd\tsplit\tseed\tval\ttest\tbest_epoch\n"
)


def run_shard(args, shard_id: int, num_shards: int, log_path: str):
    if args.gpu is not None and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    hparam_product, cls_product, splits = _build_grid(args)

    units = list(_iter_units(args, hparam_product, cls_product, splits))
    # Shard selection: keep only units where global index % N == shard_id.
    my_units = [u for i, u in enumerate(units) if i % num_shards == shard_id]

    print(
        f"[shard {shard_id}/{num_shards}] device={device_str} "
        f"loading {args.dataset} (splits={splits})",
        flush=True,
    )
    needed_splits = sorted({u[3] for u in my_units})
    data_by_split = {s: load_dataset(args.dataset, split_idx=s) for s in needed_splits}

    total_units = len(units)
    my_total = len(my_units)
    seeds_per_unit = len(args.seeds)
    print(
        f"[shard {shard_id}/{num_shards}] units={my_total} of "
        f"{total_units}; runs={my_total * seeds_per_unit}",
        flush=True,
    )

    with open(log_path, "w") as f:
        f.write(
            f"# bvb shard={shard_id}/{num_shards} "
            f"{args.dataset} at {time.ctime()}\n"
        )
        f.write(_TSV_HEADER)

    log_f = open(log_path, "a")

    t_start = time.time()
    total_runs = 0
    for unit_idx, unit in enumerate(my_units):
        hp, cls_cfg, fn, s_idx = unit
        norm, scale, alpha, beta, steps = hp
        cls, hidden, dropout, lr, wd = cls_cfg

        data = data_by_split[s_idx]
        fn_tests = []
        for seed in args.seeds:
            cfg = RunConfig(
                dataset=args.dataset,
                normalize=norm,
                feat_scale=scale,
                step_fn=fn,
                classifier=cls,
                alpha=alpha,
                beta=beta,
                ppr_steps=steps,
                hidden=hidden,
                dropout=dropout,
                lr=lr,
                wd=wd,
                epochs=args.epochs,
                patience=args.patience,
                seed=seed,
                split_idx=s_idx,
                device=device_str,
            )
            r = run_once(cfg, data)
            fn_tests.append(r.best_test)
            total_runs += 1
            log_f.write(
                f"{fn}\t{norm}\t{scale:g}\t{alpha:g}\t{beta:g}"
                f"\t{steps}\t{cls}\t{hidden}\t{dropout:g}"
                f"\t{lr:g}\t{wd:g}\t{s_idx}\t{seed}"
                f"\t{r.best_val:.4f}\t{r.best_test:.4f}\t{r.best_epoch}\n"
            )
        log_f.flush()

        elapsed = time.time() - t_start
        mean_t = sum(fn_tests) / len(fn_tests)
        pct = 100.0 * (unit_idx + 1) / my_total
        print(
            f"[sh{shard_id} {elapsed:7.1f}s {pct:5.1f}%] "
            f"fn={fn:9s} norm={norm} sc={scale:g} a={alpha:g} "
            f"b={beta:g} st={steps} sp={s_idx} cls={cls} "
            f"mean_test={mean_t:.4f}",
            flush=True,
        )
    log_f.close()
    print(
        f"[shard {shard_id}/{num_shards}] done: {total_runs} runs in "
        f"{time.time() - t_start:.1f}s",
        flush=True,
    )


# ----------------------------------------------------------------------
# Analysis / merge
# ----------------------------------------------------------------------


_COLS = [
    "fn",
    "norm",
    "scale",
    "alpha",
    "beta",
    "steps",
    "cls",
    "hidden",
    "dropout",
    "lr",
    "wd",
    "split",
    "seed",
    "val",
    "test",
    "best_epoch",
]


def _parse_row(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(_COLS):
        return None
    d = dict(zip(_COLS, parts))
    try:
        return {
            "fn": d["fn"],
            "norm": d["norm"],
            "scale": float(d["scale"]),
            "alpha": float(d["alpha"]),
            "beta": float(d["beta"]),
            "steps": int(d["steps"]),
            "cls": d["cls"],
            "hidden": int(d["hidden"]),
            "dropout": float(d["dropout"]),
            "lr": float(d["lr"]),
            "wd": float(d["wd"]),
            "split": int(d["split"]),
            "seed": int(d["seed"]),
            "val": float(d["val"]),
            "test": float(d["test"]),
        }
    except ValueError:
        return None


def _load_rows(paths: List[str]):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line or line.startswith("#") or line.startswith("fn\t"):
                    continue
                r = _parse_row(line)
                if r is not None:
                    rows.append(r)
    return rows


def _format_table(header, rows):
    all_rows = [tuple(str(x) for x in header)]
    all_rows += [tuple(str(x) for x in r) for r in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*all_rows[0]), "  ".join("-" * w for w in widths)]
    for r in all_rows[1:]:
        lines.append(fmt.format(*r))
    return lines


def analyze(rows, out_log: str, step_fns=None):
    # results[fn][cfg_key] = {(split, seed): (val, test)}
    results = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (
            r["norm"],
            r["scale"],
            r["alpha"],
            r["beta"],
            r["steps"],
            r["cls"],
            r["hidden"],
            r["dropout"],
            r["lr"],
            r["wd"],
        )
        results[r["fn"]][key][(r["split"], r["seed"])] = (r["val"], r["test"])

    fns = step_fns if step_fns is not None else sorted(results.keys())
    fns = [fn for fn in fns if fn in results]

    def select_best(fn):
        best_key = None
        best_mean_val = -1.0
        for cfg_key, runs in results[fn].items():
            vals = [v for (v, _) in runs.values()]
            if not vals:
                continue
            mv = sum(vals) / len(vals)
            if mv > best_mean_val:
                best_mean_val = mv
                best_key = cfg_key
        return best_key, best_mean_val

    def _summarize(runs):
        tests = [t for (_, t) in runs.values()]
        per_split_acc = defaultdict(list)
        for (s_idx, _seed), (_, t) in runs.items():
            per_split_acc[s_idx].append(t)
        per_split_mean = {s: sum(v) / len(v) for s, v in per_split_acc.items()}
        per_split = list(per_split_mean.values())
        mean_t = sum(tests) / len(tests)
        if len(per_split) > 1:
            m = sum(per_split) / len(per_split)
            var = sum((x - m) ** 2 for x in per_split) / (len(per_split) - 1)
            se = math.sqrt(var / len(per_split))
        else:
            se = float("nan")
        return tests, per_split_mean, mean_t, se

    def _key_row(fn, key, mv, mean_t, se, n):
        return (
            fn,
            key[0],
            f"{key[1]:g}",
            f"{key[2]:g}",
            f"{key[3]:g}",
            str(key[4]),
            key[5],
            str(key[6]),
            f"{key[7]:g}",
            f"{key[8]:g}",
            f"{key[9]:g}",
            f"{mv:.4f}",
            f"{mean_t:.4f}",
            f"{se:.4f}",
            str(n),
        )

    def _scored(fn):
        scored = []
        for cfg_key, runs in results[fn].items():
            vals = [v for (v, _) in runs.values()]
            if not vals:
                continue
            tests, _psm, mean_t, se = _summarize(runs)
            mean_v = sum(vals) / len(vals)
            scored.append((cfg_key, runs, mean_v, mean_t, se, len(tests)))
        return scored

    top_k = 5
    out = []
    out.append("\n=== TOP-5 PER VARIANT (selected on mean val acc) ===")
    header = (
        "fn",
        "norm",
        "scale",
        "alpha",
        "beta",
        "steps",
        "cls",
        "hidden",
        "dropout",
        "lr",
        "wd",
        "mean_val",
        "mean_test",
        "se_test",
        "n",
    )
    table_rows = []

    selected = {}
    fn_per_split = {}
    for fn in fns:
        scored = _scored(fn)
        if not scored:
            continue
        scored.sort(key=lambda x: x[2], reverse=True)
        for i, (cfg_key, runs, mean_v, mean_t, se, n) in enumerate(scored[:top_k]):
            table_rows.append(
                _key_row(fn if i == 0 else "", cfg_key, mean_v, mean_t, se, n)
            )
            if i == 0:
                _, per_split_mean, _, _ = _summarize(runs)
                selected[fn] = cfg_key
                fn_per_split[fn] = sorted(per_split_mean.items())
    out.extend(_format_table(header, table_rows))

    out.append("\n=== TOP-5 PER VARIANT (selected on mean test acc) ===")
    header_t = (
        "fn",
        "norm",
        "scale",
        "alpha",
        "beta",
        "steps",
        "cls",
        "hidden",
        "dropout",
        "lr",
        "wd",
        "mean_val",
        "mean_test",
        "se_test",
        "n",
    )
    test_rows = []
    for fn in fns:
        scored = _scored(fn)
        if not scored:
            continue
        scored.sort(key=lambda x: x[3], reverse=True)
        for i, (cfg_key, _runs, mean_v, mean_t, se, n) in enumerate(scored[:top_k]):
            test_rows.append(
                _key_row(fn if i == 0 else "", cfg_key, mean_v, mean_t, se, n)
            )
    out.extend(_format_table(header_t, test_rows))

    out.append("\n=== PAIRED COMPARISON vs linear (none) ===")
    if "none" not in selected:
        out.append("No 'none' baseline present; skipping paired comparison.")
    else:
        linear_by_split = dict(fn_per_split["none"])
        header2 = (
            "fn",
            "mean_test_fn",
            "mean_test_linear",
            "delta",
            "se_delta",
            "t",
            "n_splits",
            "sig_2sigma",
        )
        cmp_rows = []
        for fn in fns:
            if fn == "none" or fn not in selected:
                continue
            by_split = dict(fn_per_split[fn])
            diffs = []
            for s_idx in sorted(linear_by_split):
                if s_idx in by_split:
                    diffs.append(by_split[s_idx] - linear_by_split[s_idx])
            mean_d, se_d, t = _paired_stats(diffs)
            mean_fn = sum(v for _, v in fn_per_split[fn]) / len(fn_per_split[fn])
            mean_lin = sum(v for _, v in fn_per_split["none"]) / len(
                fn_per_split["none"]
            )
            sig = "YES" if (abs(t) >= 2 and mean_d > 0) else "no"
            cmp_rows.append(
                (
                    fn,
                    f"{mean_fn:.4f}",
                    f"{mean_lin:.4f}",
                    f"{mean_d:+.4f}",
                    f"{se_d:.4f}",
                    f"{t:+.2f}",
                    str(len(diffs)),
                    sig,
                )
            )
        out.extend(_format_table(header2, cmp_rows))

    text = "\n".join(out) + "\n"
    print(text, flush=True)
    with open(out_log, "a") as f:
        f.write(text)


# ----------------------------------------------------------------------
# Single-process entry point (wraps shard=0/num_shards=1 + analyze)
# ----------------------------------------------------------------------


def sweep_single(args):
    run_shard(args, shard_id=0, num_shards=1, log_path=args.log)
    rows = _load_rows([args.log])
    analyze(rows, out_log=args.log, step_fns=args.step_fns)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="actor")
    p.add_argument("--normalizations", nargs="+", default=["none", "std", "row"])
    p.add_argument("--feat-scales", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7])
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--steps-list", type=int, nargs="+", default=[2, 5, 20])
    p.add_argument(
        "--step-fns",
        nargs="+",
        default=[
            "none",
            "tanh",
            "clamp",
            "clamp01",
            "stanh2",
            "stanh5",
            "htanh0.5",
            "htanh2",
            "soft0.05",
            "soft0.1",
        ],
    )
    p.add_argument("--classifiers", nargs="+", default=["linear"])
    p.add_argument("--hiddens", type=int, nargs="+", default=[128])
    p.add_argument("--dropouts", type=float, nargs="+", default=[0.5])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.01])
    p.add_argument("--wds", type=float, nargs="+", default=[5e-4, 5e-3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--splits", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--log", default="/tmp/bvb.log")
    # Parallelism
    p.add_argument(
        "--shard",
        type=int,
        default=None,
        help="If set, run only shard `i` of --num-shards.",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index this shard should run on. "
        "Defaults to the current default device.",
    )
    p.add_argument(
        "--merge",
        nargs="+",
        default=None,
        help="Merge mode: list of shard TSV log paths to analyze. "
        "No runs are executed.",
    )
    args = p.parse_args()

    if args.merge is not None:
        rows = _load_rows(args.merge)
        # Truncate merged summary file.
        with open(args.log, "w") as f:
            f.write(f"# bvb merge of {len(args.merge)} shards " f"at {time.ctime()}\n")
        analyze(rows, out_log=args.log, step_fns=args.step_fns)
        return

    if args.shard is not None:
        shard_log = f"{args.log}.shard{args.shard}"
        run_shard(args, args.shard, args.num_shards, log_path=shard_log)
        return

    sweep_single(args)


if __name__ == "__main__":
    main()
