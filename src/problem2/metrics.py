import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset import inverse_od_flow


def _assign_flow_bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """将客流映射到箱编号 0..len(edges)-2（edges 为训练集分位点，单调）。"""
    b = np.searchsorted(edges, x, side="right") - 1
    return np.clip(b, 0, len(edges) - 2)


def evaluate_metrics(
    model,
    x,
    y,
    scaler,
    n_nodes,
    use_log1p,
    flow_eps,
    acc_rel_taus,
    device,
    flow_bin_edges=None,
):
    """在原始客流空间计算 RMSE/MAE/R²/MAPE/WAPE 与相对误差命中率 acc_rel（非零掩码）。

    acc_rel[τ]：在满足 y>flow_eps 的格点上，|ŷ-y|/y < τ 所占比例。
    这与「分类准确率」不同；在稀疏、高噪声 OD 回归中，τ=0.2 往往只有百分之十几到三十。
    """
    if len(x) == 0:
        nan = float("nan")
        out = {
            "rmse": nan,
            "mae": nan,
            "r2": nan,
            "mape": nan,
            "wape": nan,
            "acc_rel": {},
            "acc_round_nz": nan,
            "acc_bin": nan,
        }
        for t in acc_rel_taus:
            out["acc_rel"][float(t)] = nan
        return out
    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).cpu().numpy()
        trues = y.cpu().numpy()

    preds_inv = inverse_od_flow(preds.reshape(-1, 1), scaler, use_log1p).reshape(
        -1, n_nodes ** 2
    )
    trues_inv = inverse_od_flow(trues.reshape(-1, 1), scaler, use_log1p).reshape(
        -1, n_nodes ** 2
    )
    preds_inv = np.maximum(preds_inv, 0)

    rmse = float(np.sqrt(mean_squared_error(trues_inv, preds_inv)))
    mae = float(mean_absolute_error(trues_inv, preds_inv))
    r2 = float(r2_score(trues_inv, preds_inv))

    nz = trues_inv > flow_eps
    acc_rel = {}
    if np.any(nz):
        abs_err = np.abs(preds_inv[nz] - trues_inv[nz])
        mape = float(np.mean(abs_err / np.maximum(trues_inv[nz], flow_eps)))
        wape = float(np.sum(abs_err) / np.maximum(np.sum(trues_inv[nz]), flow_eps))
        rel_err = abs_err / np.maximum(trues_inv[nz], flow_eps)
        for t in acc_rel_taus:
            acc_rel[float(t)] = float(np.mean(rel_err < float(t)))
        acc_round_nz = float(
            np.mean(np.rint(preds_inv[nz]) == np.rint(trues_inv[nz]))
        )
    else:
        mape = float("nan")
        wape = float("nan")
        for t in acc_rel_taus:
            acc_rel[float(t)] = float("nan")
        acc_round_nz = float("nan")

    acc_bin = float("nan")
    if flow_bin_edges is not None and len(flow_bin_edges) >= 2:
        bt = _assign_flow_bin(trues_inv.ravel(), flow_bin_edges)
        bp = _assign_flow_bin(preds_inv.ravel(), flow_bin_edges)
        acc_bin = float(np.mean(bt == bp))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "wape": wape,
        "acc_rel": acc_rel,
        "acc_round_nz": acc_round_nz,
        "acc_bin": acc_bin,
    }


def combined_score(m):
    """与 main 中一致：越小越好。"""
    return (m["rmse"] + m["mae"] + (1.0 - m["r2"])) / 3.0
