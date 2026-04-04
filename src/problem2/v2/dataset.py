"""
v2 数据：时间轴拼接（前三周训 / 第四周测）、log1p+StandardScaler、时间周期特征。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import option as cfg


def _build_time_features(df_times: pd.DataFrame) -> np.ndarray:
    """每行对应一个 time_slot：hour 周期 + weekday 周期 + weekend。"""
    h = df_times["hour"].values + df_times["minute"].values / 60.0
    ang_h = 2.0 * np.pi * h / 24.0
    wd = df_times["weekday"].values.astype(np.float64)
    ang_w = 2.0 * np.pi * wd / 7.0
    we = df_times["is_weekend"].values.astype(np.float64)
    return np.column_stack(
        [np.sin(ang_h), np.cos(ang_h), np.sin(ang_w), np.cos(ang_w), we]
    ).astype(np.float32)


def prep_data(train_csv: str, test_csv: str, seq_len: int):
    print("Loading and preparing data (v2: concat timeline + time feats + log1p)...")
    df_tr = pd.read_csv(train_csv, parse_dates=["time_slot"])
    df_te = pd.read_csv(test_csv, parse_dates=["time_slot"])

    nodes = sorted(set(df_tr["in_station"]) | set(df_tr["out_station"]))
    n_nodes = len(nodes)
    n_map = {n: i for i, n in enumerate(nodes)}

    times_tr = sorted(df_tr["time_slot"].unique())
    times_te = sorted(df_te["time_slot"].unique())
    all_times = sorted(set(times_tr) | set(times_te))
    T_tr = len(times_tr)
    T = len(all_times)
    t_map = {t: i for i, t in enumerate(all_times)}

    if times_tr != all_times[:T_tr] or times_te != all_times[T_tr:]:
        print(
            "Warning: train/test 时间轴与拼接假设不一致，请检查 CSV；仍按排序后拼接处理。",
        )

    flow = np.zeros((T, n_nodes, n_nodes), dtype=np.float64)
    for df in (df_tr, df_te):
        for row in df[["time_slot", "in_station", "out_station", "flow"]].itertuples(
            index=False
        ):
            ti = t_map[row[0]]
            flow[ti, n_map[row[1]], n_map[row[2]]] = row[3]

    meta = pd.DataFrame({"time_slot": all_times})
    m1 = df_tr.drop_duplicates("time_slot")[["time_slot", "hour", "minute", "weekday", "is_weekend"]]
    m2 = df_te.drop_duplicates("time_slot")[["time_slot", "hour", "minute", "weekday", "is_weekend"]]
    meta = meta.merge(pd.concat([m1, m2], ignore_index=True), on="time_slot", how="left")
    time_feats = _build_time_features(meta)

    log_flow = np.log1p(np.maximum(flow, 0.0))

    scaler_y = StandardScaler()
    scaler_y.fit(log_flow[:T_tr].reshape(-1, 1))
    flow_scaled = scaler_y.transform(log_flow.reshape(-1, 1)).reshape(T, n_nodes, n_nodes)

    scaler_t = StandardScaler()
    scaler_t.fit(time_feats[:T_tr])
    time_scaled = scaler_t.transform(time_feats).astype(np.float32)

    x_list, y_list, t_list = [], [], []
    x_te_list, y_te_list, t_te_list = [], [], []
    test_target_tidx: list[int] = []

    for i in range(T - seq_len):
        t_end = i + seq_len
        xf = flow_scaled[i:t_end]
        tf = time_scaled[i:t_end]
        yf = flow_scaled[t_end]
        if t_end < T_tr:
            x_list.append(xf)
            y_list.append(yf)
            t_list.append(tf)
        else:
            x_te_list.append(xf)
            y_te_list.append(yf)
            t_te_list.append(tf)
            test_target_tidx.append(t_end)

    def _to_tensor(xs, ys, ts):
        if not xs:
            return None, None, None
        return (
            torch.from_numpy(np.stack(xs, axis=0)).float(),
            torch.from_numpy(np.stack(ys, axis=0)).float(),
            torch.from_numpy(np.stack(ts, axis=0)).float(),
        )

    x_tr, y_tr, t_tr = _to_tensor(x_list, y_list, t_list)
    x_te, y_te, t_te = _to_tensor(x_te_list, y_te_list, t_te_list)

    meta_out = {
        "all_times": all_times,
        "T_tr": T_tr,
        "seq_len": seq_len,
        "n_nodes": n_nodes,
        "nodes": nodes,
        "test_target_tidx": np.array(test_target_tidx, dtype=np.int64),
    }
    return x_tr, y_tr, t_tr, x_te, y_te, t_te, scaler_y, scaler_t, meta_out


def get_adj(adj_file: str, n_nodes: int) -> torch.Tensor:
    if not cfg.USE_GRAPH:
        print("【消融】USE_GRAPH=False，使用单位阵（仅自环）")
        return torch.eye(n_nodes, dtype=torch.float32)

    p = Path(adj_file)
    df_adj = pd.read_csv(p, index_col=0)
    df_adj = df_adj.sort_index(axis=0).sort_index(axis=1)
    A_base = df_adj.values.astype(np.float32)

    A_multihop = np.copy(A_base)
    A_temp = np.copy(A_base)
    for _ in range(2, cfg.k_hops + 1):
        A_temp = A_temp @ A_base
        A_multihop += A_temp

    A_multihop = (A_multihop > 0).astype(np.float32)
    np.fill_diagonal(A_multihop, 1.0)

    d = np.sum(A_multihop, axis=1)
    d_inv = np.power(d, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0
    D_mat = np.diag(d_inv)
    A_norm = D_mat @ A_multihop @ D_mat
    return torch.from_numpy(A_norm.astype(np.float32))


def inverse_flow(scaler_y: StandardScaler, arr: np.ndarray) -> np.ndarray:
    """arr: scaled log1p 空间 -> 原始 flow（非负）"""
    flat = scaler_y.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
    return np.maximum(np.expm1(flat), 0.0)
