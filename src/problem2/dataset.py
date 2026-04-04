import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import option


def _apply_log1p(mat: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(mat, 0.0).astype(np.float64)).astype(np.float32)


def inverse_od_flow(scaled_flat: np.ndarray, scaler: MinMaxScaler, use_log1p: bool) -> np.ndarray:
    """将 scaler 输出还原为原始客流（若 use_log1p 则先 inverse 再 expm1）。"""
    inv = scaler.inverse_transform(scaled_flat.reshape(-1, 1)).ravel()
    if use_log1p:
        inv = np.expm1(np.maximum(inv, 0.0))
    return np.maximum(inv, 0.0)


def prep_data(train_csv, test_csv, seq_len, val_days=None, use_log1p=None):
    """
    连续时间轴：train（前 21 天）与 test（第 4 周）拼接后构造序列。
    - 训练样本：标签时间下标 y_idx < train_end_idx（不含验证窗）。
    - 验证样本：train_end_idx <= y_idx < T_train。
    - 测试样本：T_train <= y_idx < T_total（输入可含第三周末尾）。
    """
    if val_days is None:
        val_days = option.VAL_DAYS
    if use_log1p is None:
        use_log1p = option.USE_LOG1P

    print("Loading and preparing data...")
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    nodes = sorted(list(set(df_tr["in_station"]) | set(df_tr["out_station"])))
    n_nodes = len(nodes)
    n_map = {n: i for i, n in enumerate(nodes)}

    def build_tensor(df):
        df = df.copy()
        df["time_slot"] = pd.to_datetime(df["time_slot"])
        times = sorted(df["time_slot"].unique())
        t_map = {t: i for i, t in enumerate(times)}
        res = np.zeros((len(times), n_nodes, n_nodes), dtype=np.float32)
        for row in df[["time_slot", "in_station", "out_station", "flow"]].values:
            res[t_map[row[0]], n_map[row[1]], n_map[row[2]]] = row[3]
        return res

    mat_tr = build_tensor(df_tr)
    mat_te = build_tensor(df_te)
    mat_tr_raw = mat_tr.copy()

    if use_log1p:
        mat_tr = _apply_log1p(mat_tr)
        mat_te = _apply_log1p(mat_te)
        print("  [预处理] 已使用 log1p(flow) 后再归一化")

    scaler = MinMaxScaler()
    tr_scaled = scaler.fit_transform(mat_tr.reshape(-1, 1)).reshape(mat_tr.shape)
    te_scaled = scaler.transform(mat_te.reshape(-1, 1)).reshape(mat_te.shape)

    T_train = tr_scaled.shape[0]
    T_test = te_scaled.shape[0]
    full = np.concatenate([tr_scaled, te_scaled], axis=0)
    T_total = full.shape[0]

    slots_per_day = T_train / 21.0
    val_slots = int(round(val_days * slots_per_day))
    val_slots = max(val_slots, seq_len + 1)
    train_end_idx = T_train - val_slots
    if train_end_idx <= seq_len:
        raise ValueError(
            f"val_days={val_days} 过大或 seq_len={seq_len} 过大，导致无可用训练窗；请减小 VAL_DAYS 或 SEQ_LEN"
        )

    def collect_indices(i_lo, i_hi):
        """i 为窗口起点，满足 i_hi 为最后一个包含的起点（含）。"""
        xs, ys = [], []
        for i in range(i_lo, i_hi + 1):
            if i < 0 or i + seq_len >= T_total:
                continue
            xs.append(full[i : i + seq_len])
            ys.append(full[i + seq_len])
        if not xs:
            return (
                torch.zeros(0, seq_len, n_nodes, n_nodes),
                torch.zeros(0, n_nodes, n_nodes),
            )
        return torch.FloatTensor(np.stack(xs)), torch.FloatTensor(np.stack(ys))

    # 训练：seq_len <= y_idx <= train_end_idx - 1  =>  i in [0, train_end_idx - seq_len - 1]
    i_train_hi = train_end_idx - seq_len - 1
    x_train, y_train = collect_indices(0, i_train_hi)

    # 验证：train_end_idx <= y_idx <= T_train - 1  =>  i in [train_end_idx - seq_len, T_train - seq_len - 1]
    i_val_lo = train_end_idx - seq_len
    i_val_hi = T_train - seq_len - 1
    x_val, y_val = collect_indices(i_val_lo, i_val_hi)

    # 测试：T_train <= y_idx <= T_total - 1  =>  i in [T_train - seq_len, T_total - seq_len - 1]
    i_te_lo = T_train - seq_len
    i_te_hi = T_total - seq_len - 1
    x_test, y_test = collect_indices(i_te_lo, i_te_hi)

    print(
        f"  时间步: T_train={T_train}, T_test={T_test}, seq_len={seq_len}, "
        f"train_end_idx={train_end_idx} (前约 {21 - val_days:g} 天拟合训练 / 末 {val_days} 天验证)"
    )
    print(f"  样本数: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    # 训练集原始客流分位数边界，供「分档准确率」使用（与 inverse_od_flow 后量纲一致）
    pos = mat_tr_raw.ravel()
    pos = pos[pos > option.FLOW_EPS]
    K = option.ACC_BIN_K
    flow_bin_edges = None
    if pos.size >= K + 1:
        qs = np.linspace(0.0, 1.0, K + 1)
        flow_bin_edges = np.unique(np.quantile(pos.astype(np.float64), qs).astype(np.float32))
        if len(flow_bin_edges) < 2:
            flow_bin_edges = None

    meta = {
        "use_log1p": use_log1p,
        "T_train": T_train,
        "train_end_idx": train_end_idx,
        "flow_bin_edges": flow_bin_edges,
    }
    return x_train, y_train, x_val, y_val, x_test, y_test, scaler, n_nodes, meta


def get_adj(adj_file, n_nodes):
    """
    多跳可达邻接 + 对称归一化。USE_GRAPH=False 时退化为单位阵（纯时序基线）。
    """
    if not option.USE_GRAPH:
        print("【消融实验】已屏蔽拓扑图，当前为纯时序 Baseline (A=I)")
        return torch.eye(n_nodes)

    df_adj = pd.read_csv(adj_file, index_col=0)
    df_adj = df_adj.sort_index(axis=0).sort_index(axis=1)
    A_base = df_adj.values.astype(np.float32)

    A_multihop = np.copy(A_base)
    A_temp = np.copy(A_base)
    for _ in range(2, option.k_hops + 1):
        A_temp = np.matmul(A_temp, A_base)
        A_multihop += A_temp

    A_multihop = (A_multihop > 0).astype(np.float32)
    np.fill_diagonal(A_multihop, 1.0)

    D = np.sum(A_multihop, axis=1)
    D_inv = np.power(D, -0.5)
    D_inv[np.isinf(D_inv)] = 0.0
    D_mat = np.diag(D_inv)
    A_norm = D_mat @ A_multihop @ D_mat
    return torch.FloatTensor(A_norm.astype(np.float32))
