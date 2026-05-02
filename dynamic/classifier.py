"""MLP classifier head trained on propagated features z."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainResult:
    best_val: float
    best_test: float
    best_epoch: int
    train_time: float


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


def train_mlp(
    z: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    num_classes: int,
    *,
    hidden: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    wd: float = 0.0,
    epochs: int = 300,
    patience: int = 50,
    seed: int = 0,
    device: str = "cuda",
    batch_size: int = 0,
) -> TrainResult:
    """Train an MLP on z and report best-val / corresponding-test.

    z is [N, F] float (numpy or torch). labels is [N] int. train/val/test_idx
    are 1-D integer arrays.
    """
    import time

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    use_batches = batch_size > 0

    if use_batches:
        # Keep the full dataset on host memory. Each step moves only the
        # current rows to the target device, which is required for large z.
        #
        # Gather host rows with NumPy instead of torch's CPU advanced indexing.
        # On very large propagated matrices (for example SBM-500K with
        # nonlinear features), torch can hit an internal IndexKernel assert
        # while evaluating z_t[idx].  NumPy row gathering produces a normal
        # dense batch copy, then torch only has to move that batch to `dev`.
        z_np = np.asarray(z, dtype=np.float32)
        if not z_np.flags.c_contiguous:
            z_np = np.ascontiguousarray(z_np)
        y_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        tr = torch.as_tensor(train_idx, dtype=torch.long).reshape(-1)
        va = torch.as_tensor(val_idx, dtype=torch.long).reshape(-1)
        te = torch.as_tensor(test_idx, dtype=torch.long).reshape(-1)
    else:
        z_t = torch.as_tensor(z, dtype=torch.float32, device=dev)
        y_t = torch.as_tensor(labels, dtype=torch.long, device=dev)
        tr = torch.as_tensor(train_idx, dtype=torch.long, device=dev).reshape(-1)
        va = torch.as_tensor(val_idx, dtype=torch.long, device=dev).reshape(-1)
        te = torch.as_tensor(test_idx, dtype=torch.long, device=dev).reshape(-1)

    in_channels = z_np.shape[1] if use_batches else z_t.size(1)
    model = MLP(in_channels, hidden, num_classes, num_layers, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _batch(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if use_batches:
            rows = idx.cpu().numpy()
            return (
                torch.from_numpy(z_np[rows]).to(dev, non_blocking=True),
                torch.from_numpy(y_np[rows]).to(dev, non_blocking=True),
            )
        return (
            z_t[idx].to(dev, non_blocking=True),
            y_t[idx].to(dev, non_blocking=True),
        )

    def _accuracy(idx: torch.Tensor) -> float:
        correct, total = 0, 0
        for start in range(0, idx.numel(), batch_size):
            batch_idx = idx[start : start + batch_size]
            xb, yb = _batch(batch_idx)
            pred = model(xb).argmax(-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        return correct / total

    best_val, best_test, best_epoch = -1.0, 0.0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        if use_batches:
            perm = tr[torch.randperm(tr.numel())]
            for start in range(0, perm.numel(), batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = _batch(idx)
                opt.zero_grad()
                out = model(xb)
                loss = F.nll_loss(out, yb)
                loss.backward()
                opt.step()
        else:
            opt.zero_grad()
            out = model(z_t)
            loss = F.nll_loss(out[tr], y_t[tr])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            if use_batches:
                val_acc = float(_accuracy(va))
                test_acc = float(_accuracy(te))
            else:
                out = model(z_t)
                pred = out.argmax(-1)
                val_acc = float((pred[va] == y_t[va]).float().mean().item())
                test_acc = float((pred[te] == y_t[te]).float().mean().item())
            if val_acc > best_val:
                best_val, best_test, best_epoch = val_acc, test_acc, epoch
        if patience > 0 and epoch - best_epoch >= patience:
            break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return TrainResult(
        best_val=best_val,
        best_test=best_test,
        best_epoch=best_epoch,
        train_time=time.time() - t0,
    )
