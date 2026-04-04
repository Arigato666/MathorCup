"""
v2 训练脚本：Huber 损失、AdamW、学习率衰减；指标含全局与非零子集；导出预测 CSV。
"""
from __future__ import annotations

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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

import option as cfg
from dataset import get_adj, inverse_flow, prep_data
from model import N_TIME, STGATv2, STGNNv2

_HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_HERE / ".mplconfig"))
_HERE.joinpath(".mplconfig").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nz = y_true > 0
    out = {"rmse": rmse, "mae": mae, "r2": r2}
    if nz.sum() > 10:
        out["rmse_nz"] = float(np.sqrt(mean_squared_error(y_true[nz], y_pred[nz])))
        out["mae_nz"] = float(mean_absolute_error(y_true[nz], y_pred[nz]))
        out["r2_nz"] = float(r2_score(y_true[nz], y_pred[nz]))
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    scaler_y,
    n_nodes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    n = x.size(0)
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        pb = model(x[sl].to(device), t[sl].to(device)).cpu().numpy()
        preds.append(pb)
        trues.append(y[sl].numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    pred_inv = inverse_flow(scaler_y, pred.reshape(-1, 1)).reshape(-1, n_nodes ** 2)
    true_inv = inverse_flow(scaler_y, true.reshape(-1, 1)).reshape(-1, n_nodes ** 2)
    return metrics_block(true_inv, pred_inv)


def export_predictions_csv(
    path: str,
    model: nn.Module,
    x_te: torch.Tensor,
    t_te: torch.Tensor,
    y_te: torch.Tensor,
    scaler_y,
    meta: dict,
    device: torch.device,
    batch_size: int,
) -> None:
    model.eval()
    nodes: list = meta["nodes"]
    n_nodes = meta["n_nodes"]
    all_times = meta["all_times"]
    tidx = meta["test_target_tidx"]
    rows: list[tuple] = []
    n = x_te.size(0)
    off = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            pred = model(x_te[sl].to(device), t_te[sl].to(device)).cpu().numpy()
            true = y_te[sl].numpy()
            pred_inv = inverse_flow(scaler_y, pred.reshape(-1, 1)).reshape(
                -1, n_nodes, n_nodes
            )
            true_inv = inverse_flow(scaler_y, true.reshape(-1, 1)).reshape(
                -1, n_nodes, n_nodes
            )
            for b in range(pred_inv.shape[0]):
                ts = all_times[int(tidx[off + b])]
                for ii in range(n_nodes):
                    for jj in range(n_nodes):
                        rows.append(
                            (
                                ts,
                                nodes[ii],
                                nodes[jj],
                                float(pred_inv[b, ii, jj]),
                                float(true_inv[b, ii, jj]),
                            )
                        )
            off += pred_inv.shape[0]

    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["time_slot", "in_station", "out_station", "flow_pred", "flow_true"]
        )
        w.writerows(rows)
    print(f"预测已写入: {path} （共 {len(rows)} 行）")


def main() -> None:
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using device: {device}")

    pack = prep_data(cfg.TRAIN_FILE, cfg.TEST_FILE, cfg.SEQ_LEN)
    x_tr, y_tr, t_tr, x_te, y_te, t_te, scaler_y, _, meta = pack

    if x_tr is None or x_te is None:
        raise RuntimeError("训练或测试样本为空，请检查 CSV 与时间划分。")

    n_nodes = meta["n_nodes"]
    A = get_adj(cfg.ADJ_FILE, n_nodes).to(device)

    x_tr = x_tr.to(device)
    y_tr = y_tr.to(device)
    t_tr = t_tr.to(device)
    x_te = x_te.to(device)
    y_te = y_te.to(device)
    t_te = t_te.to(device)

    if cfg.MODEL == "STGNN":
        model = STGNNv2(n_nodes, cfg.HIDDEN_DIM, N_TIME, A).to(device)
    elif cfg.MODEL == "STGAT":
        model = STGATv2(n_nodes, cfg.HIDDEN_DIM, N_TIME, A).to(device)
    else:
        raise ValueError(cfg.MODEL)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=12, gamma=0.5)
    loss_fn = nn.HuberLoss(delta=1.0)

    loader = DataLoader(
        TensorDataset(x_tr, t_tr, y_tr), batch_size=cfg.BATCH_SIZE, shuffle=True
    )

    test_scores: list[float] = []
    train_scores: list[float] = []

    print("\n[Start Training v2]")
    for ep in range(cfg.EPOCHS):
        model.train()
        for xb, tb, yb in tqdm(
            loader, desc=f"Epoch {ep + 1}/{cfg.EPOCHS}", leave=False
        ):
            opt.zero_grad(set_to_none=True)
            pred = model(xb, tb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        tr = evaluate(
            model, x_tr, y_tr, t_tr, scaler_y, n_nodes, 256, device
        )
        te = evaluate(
            model, x_te, y_te, t_te, scaler_y, n_nodes, 256, device
        )
        score_tr = (tr["rmse"] + tr["mae"] + (1 - tr["r2"])) / 3.0
        score_te = (te["rmse"] + te["mae"] + (1 - te["r2"])) / 3.0
        train_scores.append(score_tr)
        test_scores.append(score_te)

        if (ep + 1) % 5 == 0 or ep == 0:
            nz_note = ""
            if "r2_nz" in te:
                nz_note = f" | R²(nz)={te['r2_nz']:.4f}"
            print(
                f"Epoch [{ep + 1}/{cfg.EPOCHS}] lr={sched.get_last_lr()[0]:.5f} | "
                f"Train {score_tr:.4f} | Test {score_te:.4f} | "
                f"RMSE={te['rmse']:.4f} MAE={te['mae']:.4f} R²={te['r2']:.4f}{nz_note}"
            )

    te_final = evaluate(
        model, x_te, y_te, t_te, scaler_y, n_nodes, 256, device
    )
    print("\n" + "=" * 52)
    print(f"v2 最终测试集（第四周目标步，输入可跨第三周末尾） MODEL={cfg.MODEL} USE_GRAPH={cfg.USE_GRAPH}")
    print("-" * 52)
    print(f"  RMSE: {te_final['rmse']:.4f}   MAE: {te_final['mae']:.4f}   R²: {te_final['r2']:.4f}")
    if "rmse_nz" in te_final:
        print(
            f"  非零真值子集  RMSE: {te_final['rmse_nz']:.4f}   MAE: {te_final['mae_nz']:.4f}   R²: {te_final['r2_nz']:.4f}"
        )
    print("=" * 52)

    # 平滑曲线（仅展示用）
    def smooth(a: list[float], k: int = 3) -> np.ndarray:
        if len(a) < k:
            return np.array(a, dtype=np.float64)
        x = np.array(a, dtype=np.float64)
        ker = np.ones(k) / k
        return np.convolve(x, ker, mode="same")

    ep = np.arange(1, cfg.EPOCHS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(ep, train_scores, alpha=0.35, label="Train combined (raw)")
    plt.plot(ep, test_scores, alpha=0.35, label="Test combined (raw)")
    plt.plot(ep, smooth(train_scores), lw=2, label="Train (moving avg)")
    plt.plot(ep, smooth(test_scores), lw=2, label="Test (moving avg)")
    plt.xlabel("Epoch")
    plt.ylabel("(RMSE + MAE + (1-R²)) / 3")
    tag = "graph" if cfg.USE_GRAPH else "nograph"
    plt.title(f"v2 learning curve ({cfg.MODEL}, {tag})")
    plt.legend()
    plt.grid(True, alpha=0.35)
    out_png = _HERE / f"learning_curve_v2_{cfg.MODEL}_{tag}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"曲线已保存: {out_png}")

    export_predictions_csv(
        cfg.EXPORT_PRED_CSV,
        model,
        x_te.cpu(),
        t_te.cpu(),
        y_te.cpu(),
        scaler_y,
        meta,
        device,
        256,
    )


if __name__ == "__main__":
    main()
