"""
ST-GCN / Pure-GRU 消融；可选加权 Huber；验证集仍为第四周（与 v3 一致，注意论文表述）。
学习曲线：双子图、原始 train/val，避免 convolve 边界伪影。
"""
from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

import option as cfg
from dataset import ODWindowDataset, load_and_build, symmetric_adj
from model import PureGRUModel, STGCNModel

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_HERE / ".mplconfig"))
_HERE.joinpath(".mplconfig").mkdir(parents=True, exist_ok=True)


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class WeightedHuberLoss(nn.Module):
    """在归一化空间：流量越大（target 越大）权重越高，减轻全零样本主导梯度。"""

    def __init__(self, delta: float = 1.0, alpha: float = 2.0):
        super().__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = 1.0 + self.alpha * target
        diff = torch.abs(pred - target)
        huber = torch.where(
            diff < self.delta,
            0.5 * diff * diff,
            self.delta * (diff - 0.5 * self.delta),
        )
        return (w * huber).mean()


def build_criterion() -> nn.Module:
    if cfg.USE_WEIGHTED_HUBER:
        return WeightedHuberLoss(delta=cfg.HUBER_DELTA, alpha=cfg.WEIGHTED_HUBER_ALPHA)
    return nn.HuberLoss(delta=cfg.HUBER_DELTA)


def run_epoch(model, loader, device, criterion, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    for x_flow, x_time, y in loader:
        x_flow = x_flow.to(device)
        x_time = x_time.to(device)
        y = y.to(device)
        pred = model(x_flow, x_time)
        loss = criterion(pred, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def train_with_early_stop(model, train_loader, val_loader, device, name: str):
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.EPOCHS, eta_min=1e-5
    )
    crit = build_criterion()
    best = float("inf")
    best_state = None
    bad = 0
    hist_tr, hist_val = [], []

    loss_label = "wHuber" if cfg.USE_WEIGHTED_HUBER else "Huber"
    for ep in range(1, cfg.EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, crit, True, opt)
        va = run_epoch(model, val_loader, device, crit, False, None)
        sched.step()
        hist_tr.append(tr)
        hist_val.append(va)
        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 10 == 0:
            print(
                f"[{name}] Epoch {ep:3d}/{cfg.EPOCHS}  train_{loss_label}={tr:.5f}  "
                f"val_{loss_label}={va:.5f}  lr={sched.get_last_lr()[0]:.6f}"
            )
        if bad >= cfg.PATIENCE:
            print(f"[{name}] Early stopping at epoch {ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist_tr, hist_val


@torch.no_grad()
def evaluate_metrics(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for x_flow, x_time, y in loader:
        p = model(x_flow.to(device), x_time.to(device)).cpu().numpy()
        preds.append(p)
        trues.append(y.numpy())
    pred_n = np.vstack(preds)
    true_n = np.vstack(trues)
    pred = scaler.inverse_transform(pred_n)
    true = scaler.inverse_transform(true_n)
    pred = np.clip(pred, 0, None)
    pf = pred.ravel()
    tf = true.ravel()
    rmse = float(np.sqrt(mean_squared_error(tf, pf)))
    mae = float(mean_absolute_error(tf, pf))
    r2 = float(r2_score(tf, pf))
    return {"rmse": rmse, "mae": mae, "r2": r2}, pred, true


@torch.no_grad()
def export_submission(
    path: str,
    model,
    loader,
    scaler,
    device,
    pairs: list[tuple[str, str]],
    all_times: list,
    tidx: np.ndarray,
):
    model.eval()
    rows = []
    k = 0
    for x_flow, x_time, y in loader:
        pred = model(x_flow.to(device), x_time.to(device)).cpu().numpy()
        pred = scaler.inverse_transform(pred)
        pred = np.clip(pred, 0, None)
        for b in range(pred.shape[0]):
            ts = all_times[int(tidx[k + b])]
            for j, (o, d) in enumerate(pairs):
                rows.append((ts, o, d, round(float(pred[b, j]), 4)))
        k += pred.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "in_station", "out_station", "flow_pred"])
        w.writerows(rows)
    print(f"提交表已保存: {path} （{len(rows)} 行）")


def plot_learning_curves(
    h_tr_s: list[float],
    h_va_s: list[float],
    h_tr_g: list[float],
    h_va_g: list[float],
    path: str,
) -> None:
    # 与 visualize_problem2 统一：ST-GCN 深紫 #7F4A88，Pure-GRU 豆沙粉 #DE95BA
    c_stgcn, c_gru = "#7F4A88", "#DE95BA"
    bg, grid_c = "#FFFDFB", "#EEE8EE"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(bg)
    loss_name = "Weighted Huber" if cfg.USE_WEIGHTED_HUBER else "Huber"
    for ax, h_tr, h_va, title, color in zip(
        axes,
        [h_tr_s, h_tr_g],
        [h_va_s, h_va_g],
        ["ST-GCN", "Pure-GRU"],
        [c_stgcn, c_gru],
    ):
        ax.set_facecolor("#FFFFFF")
        ep = np.arange(1, len(h_va) + 1)
        ax.plot(ep, h_tr, color=color, alpha=0.5, lw=1.25, label="Train")
        ax.plot(ep, h_va, color=color, lw=2.05, label="Val (week 4)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(loss_name)
        ax.set_title(title, fontweight="600")
        ax.legend(framealpha=0.95, edgecolor="#D4C4D0")
        ax.grid(True, color=grid_c, linestyle="-", linewidth=0.7, alpha=1.0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#9B8AA0")
    plt.suptitle(
        "v4: ablation learning curves (lower is better)",
        fontweight="bold",
        color="#202124",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"曲线: {path}")


def main() -> None:
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Device: {device}")
    print(
        f"损失: {'WeightedHuber' if cfg.USE_WEIGHTED_HUBER else 'Huber'} "
        f"(epochs={cfg.EPOCHS}, patience={cfg.PATIENCE})"
    )

    flow, time_f, scaler, tr_idx, te_idx, meta, L = load_and_build(
        cfg.TRAIN_CSV, cfg.TEST_CSV, cfg.SEQ_LEN
    )
    print(
        f"样本数 train={len(tr_idx)} test={len(te_idx)} | OD={meta['n_od']} | L={L}"
    )

    A_norm = symmetric_adj(cfg.ADJ_CSV).to(device)

    ds_tr = ODWindowDataset(flow, time_f, tr_idx, L)
    ds_te = ODWindowDataset(flow, time_f, te_idx, L)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=cfg.BATCH_SIZE, shuffle=False)

    nodes = meta["nodes"]
    pairs = meta["pairs"]

    stgcn = STGCNModel(
        nodes=nodes,
        n_od=meta["n_od"],
        od_pairs=pairs,
        A_norm=A_norm,
        node_emb_dim=cfg.NODE_EMB_DIM,
        gcn_dim=cfg.GCN_DIM,
        gru_hidden=cfg.GRU_HIDDEN,
        time_feat_dim=cfg.TIME_FEAT_DIM,
        seq_len=L,
    ).to(device)

    gru_only = PureGRUModel(
        n_od=meta["n_od"],
        gru_hidden=cfg.GRU_HIDDEN,
        time_feat_dim=cfg.TIME_FEAT_DIM,
        seq_len=L,
    ).to(device)

    print("\n>>> ST-GCN")
    h_tr_s, h_va_s = train_with_early_stop(stgcn, dl_tr, dl_te, device, "ST-GCN")
    print("\n>>> Pure-GRU (ablation)")
    h_tr_g, h_va_g = train_with_early_stop(gru_only, dl_tr, dl_te, device, "Pure-GRU")

    m_s, _, _ = evaluate_metrics(stgcn, dl_te, scaler, device)
    m_g, _, _ = evaluate_metrics(gru_only, dl_te, scaler, device)

    print("\n" + "=" * 58)
    print(f"{'模型':<18} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 58)
    print(f"{'ST-GCN':<18} {m_s['rmse']:>10.4f} {m_s['mae']:>10.4f} {m_s['r2']:>10.4f}")
    print(f"{'Pure-GRU':<18} {m_g['rmse']:>10.4f} {m_g['mae']:>10.4f} {m_g['r2']:>10.4f}")
    print("=" * 58)
    if m_g["rmse"] > 0:
        print(
            f"\nST-GCN 相对 Pure-GRU: RMSE ↓{(m_g['rmse'] - m_s['rmse']) / m_g['rmse'] * 100:.1f}%  "
            f"MAE ↓{(m_g['mae'] - m_s['mae']) / m_g['mae'] * 100:.1f}%  "
            f"R² ↑{m_s['r2'] - m_g['r2']:.4f}"
        )

    plot_learning_curves(h_tr_s, h_va_s, h_tr_g, h_va_g, cfg.OUT_CURVE)

    export_submission(
        cfg.OUT_SUBMISSION,
        stgcn,
        dl_te,
        scaler,
        device,
        pairs,
        meta["all_times"],
        meta["test_target_tidx"],
    )
    export_submission(
        cfg.OUT_SUBMISSION_GRU,
        gru_only,
        dl_te,
        scaler,
        device,
        pairs,
        meta["all_times"],
        meta["test_target_tidx"],
    )


if __name__ == "__main__":
    main()
