"""Nonlinear PageRank dynamic maintenance.

The C++ extension implementing Algorithm 1 from the writeup is JIT-compiled
on first import via torch.utils.cpp_extension.load. Build outputs are cached
under ~/.cache/torch_extensions/.
"""

import os
import sys
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_BIN = os.path.dirname(sys.executable)
if _PY_BIN and _PY_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PY_BIN + os.pathsep + os.environ.get("PATH", "")


def _openmp_flags():
    return ["-fopenmp"], ["-fopenmp"]


def _build_extension(verbose: bool = False):
    cflags_omp, ldflags_omp = _openmp_flags()
    return load(
        name="nl_ppr",
        sources=[
            os.path.join(_HERE, "csrc", "bindings.cpp"),
            os.path.join(_HERE, "csrc", "nonlinear_ppr.cpp"),
        ],
        extra_include_paths=[os.path.join(_HERE, "csrc")],
        extra_cflags=["-O3", "-std=c++17", *cflags_omp, "-DNDEBUG"],
        extra_ldflags=ldflags_omp,
        verbose=verbose,
    )


_ext = _build_extension(verbose=bool(int(os.environ.get("NLPPR_BUILD_VERBOSE", "0"))))

NonlinearPPR = _ext.NonlinearPPR
StepFn = _ext.StepFn
ThresholdMode = _ext.ThresholdMode

THRESHOLD_MODE_NAMES = {
    "degree": ThresholdMode.DEGREE,
    "rowsum": ThresholdMode.ROWSUM,
}


def resolve_threshold_mode(name: str):
    key = name.lower()
    if key not in THRESHOLD_MODE_NAMES:
        raise ValueError(
            f"Unknown threshold_mode: {name}. Known: " f"{list(THRESHOLD_MODE_NAMES)}"
        )
    return THRESHOLD_MODE_NAMES[key]


STEP_FN_NAMES = {
    "none": StepFn.IDENTITY,
    "identity": StepFn.IDENTITY,
    "sigmoid": StepFn.SIGMOID,
    "tanh": StepFn.TANH,
    "softplus": StepFn.SOFTPLUS,
    "relu": StepFn.RELU,
    "clamp": StepFn.CLAMP_SYM,
    "stanh": StepFn.STANH,
    "leaky": StepFn.LEAKY05,
    "shtanh": StepFn.SHTANH,
    "htanh": StepFn.HTANH,
}

# Lipschitz constant K for each step function. Used to set the algorithm's
# contraction factor; affects the push threshold (1 - K*(1-alpha))*eps*d^{1-beta}.
STEP_FN_K = {
    "none": 1.0,
    "identity": 1.0,
    "sigmoid": 0.25,
    "tanh": 1.0,
    "softplus": 1.0,
    "relu": 1.0,
    "clamp": 1.0,
    "stanh": 1.0,
    "leaky": 1.0,
    "shtanh": 1.0,
    "htanh": 1.0,
}


def resolve_step_fn(name: str):
    """Return (StepFn enum, default Lipschitz K) for a given step function name."""
    key = name.lower()
    if key not in STEP_FN_NAMES:
        raise ValueError(f"Unknown step_fn: {name}. Known: {list(STEP_FN_NAMES)}")
    return STEP_FN_NAMES[key], STEP_FN_K[key]


__all__ = [
    "NonlinearPPR",
    "StepFn",
    "STEP_FN_NAMES",
    "STEP_FN_K",
    "resolve_step_fn",
    "ThresholdMode",
    "THRESHOLD_MODE_NAMES",
    "resolve_threshold_mode",
]
