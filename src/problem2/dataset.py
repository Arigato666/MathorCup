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

    # 铺平做归一化
    scaler = MinMaxScaler()
    tr_scaled = scaler.fit_transform(mat_tr.reshape(-1, n_nodes ** 2)).reshape(mat_tr.shape)
    te_scaled = scaler.transform(mat_te.reshape(-1, n_nodes ** 2)).reshape(mat_te.shape)

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
    # 如果消融实验关掉了图结构，直接返回单位矩阵 (退化为不考虑邻居)
    if not option.USE_GRAPH:
        print("【消融实验】已屏蔽拓扑图，当前为纯时序 Baseline")
        return torch.eye(n_nodes)

    df_adj = pd.read_csv(adj_file, index_col=0)
    df_adj = df_adj.sort_index(axis=0).sort_index(axis=1)
    A = df_adj.values.astype(np.float32)

    # GCN 归一化: D^-0.5 * (A+I) * D^-0.5
    np.fill_diagonal(A, 1.0)
    D = np.sum(A, axis=1)
    D_inv = np.power(D, -0.5)
    D_inv[np.isinf(D_inv)] = 0.
    D_mat = np.diag(D_inv)

    A_norm = D_mat @ A @ D_mat
    return torch.FloatTensor(A_norm)