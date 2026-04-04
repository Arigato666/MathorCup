"""
训练 ST-GCN 与 Pure-GRU 消融；验证损失为 Huber（量纲统一）；导出第四周预测 CSV。
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
from tqdm import tqdm

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


def smooth(x: list[float], k: int = 3) -> np.ndarray:
    a = np.array(x, dtype=np.float64)
    if len(a) < k:
        return a
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


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


def train_with_early_stop(
    model,
    train_loader,
    val_loader,
    device,
    name: str,
):
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.EPOCHS, eta_min=1e-5
    )
    crit = nn.HuberLoss(delta=1.0)
    best = float("inf")
    best_state = None
    bad = 0
    hist_tr, hist_val = [], []

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
        if ep == 1 or ep % 5 == 0:
            print(
                f"[{name}] Epoch {ep:3d}/{cfg.EPOCHS}  train_hub={tr:.5f}  val_hub={va:.5f}  lr={sched.get_last_lr()[0]:.6f}"
            )
        if bad >= cfg.PATIENCE:
            print(f"[{name}] Early stopping at epoch {ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist_tr, hist_val


@torch.no_grad()
def evaluate_metrics(model, loader, scaler, device, n_od: int):
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
                rows.append((ts, o, d, float(pred[b, j])))
        k += pred.shape[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "in_station", "out_station", "flow_pred"])
        w.writerows(rows)
    print(f"提交表已保存: {path} （{len(rows)} 行）")


def main() -> None:
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Device: {device}")

    flow, time_f, scaler, tr_idx, te_idx, meta, L = load_and_build(
        cfg.TRAIN_CSV, cfg.TEST_CSV, cfg.SEQ_LEN
    )
    print(
        f"样本数 train={len(tr_idx)} test={len(te_idx)} | OD={meta['n_od']} | L={L}"
    )

    A_norm = symmetric_adj(cfg.ADJ_CSV).to(device)

    ds_tr = ODWindowDataset(flow, time_f, tr_idx, L)
    ds_te = ODWindowDataset(flow, time_f, te_idx, L)
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False
    )
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

    m_s, _, _ = evaluate_metrics(stgcn, dl_te, scaler, device, meta["n_od"])
    m_g, _, _ = evaluate_metrics(gru_only, dl_te, scaler, device, meta["n_od"])

    print("\n" + "=" * 58)
    print(f"{'模型':<18} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 58)
    print(f"{'ST-GCN':<18} {m_s['rmse']:>10.4f} {m_s['mae']:>10.4f} {m_s['r2']:>10.4f}")
    print(f"{'Pure-GRU':<18} {m_g['rmse']:>10.4f} {m_g['mae']:>10.4f} {m_g['r2']:>10.4f}")
    print("=" * 58)

    ep = np.arange(1, len(h_va_s) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(ep, h_va_s, alpha=0.4, label="ST-GCN val Huber (raw)")
    plt.plot(ep, smooth(h_va_s), lw=2, label="ST-GCN val (mov. avg)")
    plt.plot(ep, h_va_g, alpha=0.4, label="Pure-GRU val Huber (raw)")
    plt.plot(ep, smooth(h_va_g), lw=2, label="Pure-GRU val (mov. avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Huber loss")
    plt.title("v3: ablation learning curves (lower is better)")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(cfg.OUT_CURVE, dpi=200)
    print(f"曲线: {cfg.OUT_CURVE}")

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


if __name__ == "__main__":
    main()
