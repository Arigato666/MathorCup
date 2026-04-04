"""
与 v3 相同的全时间轴 + 跨第三周/第四周边界的输入窗口。

说明（纠正常见误判）：
- MinMaxScaler 仅在 mat[:T_tr] 上 fit，测试段只做 transform，不存在「用测试集统计量 fit」。
- 曾出现的 RMSE=0、R²=1 来自时间戳类型混用导致 t_map 查表失败、矩阵全零，并非「把测试写进矩阵」本身。
- 若测试窗口的输入里含第四周已发生的真实 OD（teacher forcing），属于在线预测意义上的信息泄漏；
  赛题批量评估里常用此设定；若需严格滚动预测，应改用逐步递推或仅用训练段为输入的窗口（会改变可预测时刻集合）。

不把训练/测试拆成两个独立时间轴滑窗：否则第四周前 L 个时刻无法利用第三周末尾作输入，任务定义被改变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def od_pairs_56(nodes: list[str]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
    pairs = [(o, d) for o in nodes for d in nodes if o != d]
    return pairs, {p: i for i, p in enumerate(pairs)}


def build_time_features(slots: list[pd.Timestamp]) -> np.ndarray:
    feats = []
    for s in slots:
        ts = pd.Timestamp(s)
        h = ts.hour + ts.minute / 60.0
        wd = float(ts.weekday())
        feats.append(
            [
                np.sin(2 * np.pi * h / 24.0),
                np.cos(2 * np.pi * h / 24.0),
                np.sin(2 * np.pi * wd / 7.0),
                np.cos(2 * np.pi * wd / 7.0),
                1.0 if wd >= 5 else 0.0,
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def load_and_build(
    train_csv: str,
    test_csv: str,
    seq_len: int,
):
    df_tr = pd.read_csv(train_csv, parse_dates=["time_slot"])
    df_te = pd.read_csv(test_csv, parse_dates=["time_slot"])

    nodes = sorted(set(df_tr["in_station"]) | set(df_tr["out_station"]))
    pairs, od2idx = od_pairs_56(nodes)
    n_od = len(pairs)

    times_tr = sorted({pd.Timestamp(t) for t in df_tr["time_slot"].unique()})
    times_te = sorted({pd.Timestamp(t) for t in df_te["time_slot"].unique()})
    all_times = sorted(set(times_tr) | set(times_te))
    T_tr = len(times_tr)
    T = len(all_times)
    t_map = {pd.Timestamp(t): i for i, t in enumerate(all_times)}

    mat = np.zeros((T, n_od), dtype=np.float32)
    for df in (df_tr, df_te):
        cols = ["time_slot", "in_station", "out_station", "flow"]
        for row in df[cols].itertuples(index=False):
            key = (row[1], row[2])
            if key not in od2idx:
                continue
            try:
                ts = pd.Timestamp(row[0])
            except Exception:
                ts = pd.to_datetime(row[0])
            if ts not in t_map:
                continue
            mat[t_map[ts], od2idx[key]] = float(row[3])

    time_feat = build_time_features(all_times)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(mat[:T_tr])
    mat_norm = scaler.transform(mat).astype(np.float32)

    train_idx = list(range(0, T_tr - seq_len))
    test_idx = list(range(T_tr - seq_len, T - seq_len))

    meta = {
        "all_times": all_times,
        "T_tr": T_tr,
        "n_od": n_od,
        "pairs": pairs,
        "nodes": nodes,
        "test_target_tidx": np.array([i + seq_len for i in test_idx], dtype=np.int64),
    }
    return mat_norm, time_feat, scaler, train_idx, test_idx, meta, seq_len


class ODWindowDataset(Dataset):
    def __init__(
        self,
        flow: np.ndarray,
        time_feat: np.ndarray,
        start_indices: list[int],
        seq_len: int,
    ):
        self.flow = torch.from_numpy(flow)
        self.time = torch.from_numpy(time_feat)
        self.starts = start_indices
        self.L = seq_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        i = self.starts[idx]
        return (
            self.flow[i : i + self.L],
            self.time[i : i + self.L],
            self.flow[i + self.L],
        )


def symmetric_adj(adj_csv: str) -> torch.Tensor:
    df = pd.read_csv(adj_csv, index_col=0)
    df = df.sort_index(axis=0).sort_index(axis=1)
    A = df.values.astype(np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    d = A_hat.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = np.diag(d_inv_sqrt)
    A_norm = D @ A_hat @ D
    return torch.from_numpy(A_norm.astype(np.float32))
