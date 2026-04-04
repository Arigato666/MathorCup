"""v2: 在时空块中注入周期时间特征；GCN/GAT + GRU。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TIME = 5


class STGNNv2(nn.Module):
    def __init__(self, n_nodes: int, h_dim: int, n_time: int, A: torch.Tensor):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.register_buffer("A", A)

        self.W = nn.Parameter(torch.empty(n_nodes, h_dim))
        nn.init.xavier_normal_(self.W)
        self.time_fc = nn.Linear(n_time, h_dim)
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, n_nodes)

    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        b, s, n, _ = x.shape
        xw = torch.relu(x @ self.W)
        te = self.time_fc(t_feat).unsqueeze(2).expand(-1, -1, n, -1)
        h = torch.relu(xw + te)
        h = torch.relu(torch.matmul(self.A, h))
        h = h.permute(0, 2, 1, 3).reshape(b * n, s, self.h_dim)
        _, ht = self.gru(h)
        out = self.fc(ht[-1])
        return out.reshape(b, n, n)


class STGATv2(nn.Module):
    def __init__(self, n_nodes: int, h_dim: int, n_time: int, A: torch.Tensor):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.register_buffer("A_mask", (A > 0).float())

        self.W = nn.Linear(n_nodes, h_dim)
        self.time_fc = nn.Linear(n_time, h_dim)
        self.a_src = nn.Parameter(torch.empty(h_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(h_dim, 1))
        nn.init.xavier_normal_(self.a_src)
        nn.init.xavier_normal_(self.a_dst)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, n_nodes)

    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        b, s, n, _ = x.shape
        h = self.W(x)
        te = self.time_fc(t_feat).unsqueeze(2).expand(-1, -1, n, -1)
        h = h + te

        attn_src = torch.matmul(h, self.a_src).squeeze(-1)
        attn_dst = torch.matmul(h, self.a_dst).squeeze(-1)
        e = self.leakyrelu(attn_src.unsqueeze(3) + attn_dst.unsqueeze(2))
        neg = torch.full_like(e, -9e15)
        mask = self.A_mask.unsqueeze(0).unsqueeze(0).expand(b, s, n, n)
        attention = torch.where(mask > 0, e, neg)
        attention = F.softmax(attention, dim=-1)
        h_sp = torch.matmul(attention, h)
        h_sp = F.elu(h_sp) + h

        h_sp = h_sp.permute(0, 2, 1, 3).reshape(b * n, s, self.h_dim)
        _, ht = self.gru(h_sp)
        out = self.fc(ht[-1])
        return out.reshape(b, n, n)
