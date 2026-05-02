# Nonlinear PageRank experiments

This directory is a standalone, anonymized copy of the experiment code.

It contains:

- `bvb.py` and `experiment.py` for static best-vs-best node-classification
  sweeps.
- `dynamic/` for dynamic graph experiments.
- `scripts/run_static_bvb.sh` for the static sweeps.
- `scripts/run_dynamic.sh` for the dynamic main, epsilon, and alpha sweeps.

## Setup

Use Python 3.12 with the packages in `requirements.txt`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The runners use `${PYTHON}` when it is set. Otherwise they prefer
`.venv/bin/python` and fall back to `python3`.

## Data

PyG and OGB datasets are downloaded into `data/` by their loaders.

For the dynamic SBM-500K experiment, place the SBM text files under
`data/sbm-500k/`. See `dynamic/README.md` for the expected filenames.

## Running

Static BVB sweeps:

```bash
scripts/run_static_bvb.sh
```

Dynamic sweeps:

```bash
scripts/run_dynamic.sh
```

Both scripts are local-process launchers. Increase parallelism with
`MAX_JOBS`, select GPUs with `GPUS`, and reduce run size for smoke tests with
the documented environment variables inside each script.
