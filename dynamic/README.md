# Dynamic experiments

This package contains the dynamic nonlinear PageRank maintenance code used by
`scripts/run_dynamic.sh`.

The C++ extension is JIT-compiled on first import through
`torch.utils.cpp_extension.load`; build products are cached by PyTorch outside
this repository. Set `NLPPR_BUILD_VERBOSE=1` to print compiler commands.

## Data defaults

- OGB datasets are read from `data/ogb`.
- SBM-500K files are read from `data/sbm-500k`. For generating or obtaining
  them, refer to <https://github.com/zheng-yp/InstantGNN>.

The SBM directory should contain:

```text
SBM-500000-50-20+1_init.txt
SBM-500000-50-20+1_label.txt
SBM-500000-50-20+1_Edgeupdate_snap0.txt
...
SBM-500000-50-20+1_Edgeupdate_snap9.txt
SBM-500000-50-20+1_label_snap0.txt
...
SBM-500000-50-20+1_label_snap9.txt
```

## Direct run example

```bash
OMP_NUM_THREADS=16 python -m dynamic.run_experiment \
  --dataset arxiv \
  --mode dynamic_batched \
  --step-fn htanh --step-param 2.5 \
  --feat-scale 3 -t rowsum --alpha 0.1 --beta 0.5 \
  --eps 1e-7 --mlp-epochs 1000 \
  --mlp-hidden 1024 --mlp-num-layers 4 \
  --out outputs/dynamic/arxiv/nl0.json
```
