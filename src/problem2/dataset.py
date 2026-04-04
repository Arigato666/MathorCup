import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import option


def prep_data(train_csv, test_csv, seq_len):
    print("Loading and preparing data...")
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    # 抽取所有节点并排序映射
    nodes = sorted(list(set(df_tr['in_station']) | set(df_tr['out_station'])))
    n_nodes = len(nodes)
    n_map = {n: i for i, n in enumerate(nodes)}

    def build_tensor(df):
        df['time_slot'] = pd.to_datetime(df['time_slot'])
        times = sorted(df['time_slot'].unique())
        t_map = {t: i for i, t in enumerate(times)}

        res = np.zeros((len(times), n_nodes, n_nodes))
        # 极速遍历
        for row in df[['time_slot', 'in_station', 'out_station', 'flow']].values:
            res[t_map[row[0]], n_map[row[1]], n_map[row[2]]] = row[3]
        return res

    mat_tr = build_tensor(df_tr)
    mat_te = build_tensor(df_te)

    # 铺平为单列做【全局】归一化，保留 OD 对之间的大小差异
    scaler = MinMaxScaler()
    tr_scaled = scaler.fit_transform(mat_tr.reshape(-1, 1)).reshape(mat_tr.shape)
    te_scaled = scaler.transform(mat_te.reshape(-1, 1)).reshape(mat_te.shape)

    # 切片做滑动窗口
    def make_seq(data):
        x, y = [], []
        for i in range(len(data) - seq_len):
            x.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))

    x_tr, y_tr = make_seq(tr_scaled)
    x_te, y_te = make_seq(te_scaled)

    return x_tr, y_tr, x_te, y_te, scaler, n_nodes


def get_adj(adj_file, n_nodes):
    """
    根据EDA 结论 (h=4 时流量最大)，构建高阶多跳邻接矩阵。
    让模型突破 1 跳的近视眼，直接捕捉中远距离的拓扑相关性。
    """
    if not option.USE_GRAPH:
        print("【消融实验】已屏蔽拓扑图，当前为纯时序 Baseline")
        return torch.eye(n_nodes)

    df_adj = pd.read_csv(adj_file, index_col=0)
    df_adj = df_adj.sort_index(axis=0).sort_index(axis=1)
    A_base = df_adj.values.astype(np.float32)

    # === 核心改进：计算 K 跳可达矩阵 ===
    # 既然 h=4 是主力，我们就让节点不仅连着邻居，还连着邻居的邻居...
    A_multihop = np.copy(A_base)
    A_temp = np.copy(A_base)

    for _ in range(2, option.k_hops + 1):
        A_temp = np.matmul(A_temp, A_base)  # A^2, A^3, A^4...
        A_multihop += A_temp

    # 只要 k 跳以内能到的，都在掩码图里标为 1 (连通)
    A_multihop = (A_multihop > 0).astype(np.float32)
    # ==================================

    # 加上自环
    np.fill_diagonal(A_multihop, 1.0)

    # 归一化 D^-0.5 * A * D^-0.5
    D = np.sum(A_multihop, axis=1)
    D_inv = np.power(D, -0.5)
    D_inv[np.isinf(D_inv)] = 0.
    D_mat = np.diag(D_inv)

    A_norm = D_mat @ A_multihop @ D_mat

    return torch.FloatTensor(A_norm)