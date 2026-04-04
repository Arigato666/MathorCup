import torch
import torch.nn as nn
import torch.nn.functional as F


class STGAT(nn.Module):
    """时空图注意力 (STGAT) + GRU：在拓扑掩码上做注意力，沿时间用 GRU。"""

    def __init__(self, n_nodes, h_dim, A):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim

        self.register_buffer("A_mask", (A > 0).float())

        self.W = nn.Linear(n_nodes, h_dim)
        self.a_src = nn.Parameter(torch.empty(h_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(h_dim, 1))
        nn.init.xavier_normal_(self.a_src)
        nn.init.xavier_normal_(self.a_dst)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, n_nodes)

    def forward(self, x):
        b, s, n, _ = x.shape
        h = self.W(x)

        attn_src = torch.matmul(h, self.a_src).squeeze(-1)
        attn_dst = torch.matmul(h, self.a_dst).squeeze(-1)
        e = attn_src.unsqueeze(3) + attn_dst.unsqueeze(2)
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        mask = self.A_mask.unsqueeze(0).unsqueeze(0).expand(b, s, n, n)
        attention = torch.where(mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)

        h_sp = torch.matmul(attention, h)
        h_sp = F.elu(h_sp)
        h_sp = h_sp + h

        h_sp = h_sp.permute(0, 2, 1, 3).reshape(b * n, s, self.h_dim)
        _, ht = self.gru(h_sp)
        out = self.fc(ht[-1])
        return out.reshape(b, n, n)


class STGNN(nn.Module):
    """GCN（对称归一化邻接）+ GRU。"""

    def __init__(self, n_nodes, h_dim, A):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.register_buffer("A", A.clone())

        self.W = nn.Parameter(torch.empty(n_nodes, h_dim))
        nn.init.xavier_normal_(self.W)

        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, n_nodes)

    def forward(self, x):
        b, s, n, _ = x.shape
        xw = x @ self.W
        h_sp = torch.relu(self.A @ xw)
        h_sp = h_sp.permute(0, 2, 1, 3).reshape(b * n, s, self.h_dim)
        _, ht = self.gru(h_sp)
        out = self.fc(ht[-1])
        return out.reshape(b, n, n)
