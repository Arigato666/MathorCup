import torch
import torch.nn as nn


class STGNN(nn.Module):
    def __init__(self, n_nodes, h_dim, A):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.A = A

        # 空间: GCN 权重
        self.W = nn.Parameter(torch.empty(n_nodes, h_dim))
        nn.init.xavier_normal_(self.W)

        # 时间: GRU
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)

        # 预测头
        self.fc = nn.Linear(h_dim, n_nodes)

    def forward(self, x):
        b, s, n, _ = x.shape

        # 1. 空间扩散 (图卷积)
        xw = x @ self.W
        h_sp = torch.relu(self.A @ xw)

        # 2. 时序演化 (合并维度送入GRU)
        h_sp = h_sp.permute(0, 2, 1, 3).reshape(b * n, s, self.h_dim)
        _, ht = self.gru(h_sp)

        # 3. 输出提取最后一步
        out = self.fc(ht[-1])
        return out.reshape(b, n, n)