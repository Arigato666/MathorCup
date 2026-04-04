"""ST-GCN（节点嵌入 + GCN + 按 OD 的 GRU）与纯 GRU 消融。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, A_norm: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)
        self.register_buffer("A_norm", A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        support = self.lin(x)
        A = self.A_norm.unsqueeze(0).expand(x.size(0), -1, -1)
        return F.relu(torch.bmm(A, support))


class STGCNModel(nn.Module):
    def __init__(
        self,
        nodes: list[str],
        n_od: int,
        od_pairs: list[tuple[str, str]],
        A_norm: torch.Tensor,
        node_emb_dim: int,
        gcn_dim: int,
        gru_hidden: int,
        time_feat_dim: int,
        seq_len: int,
    ):
        super().__init__()
        self.n_node = len(nodes)
        self.n_od = n_od
        self.seq_len = seq_len
        self.node2idx = {n: i for i, n in enumerate(nodes)}

        self.node_emb = nn.Embedding(self.n_node, node_emb_dim)
        self.gcn1 = GraphConvLayer(node_emb_dim, gcn_dim, A_norm)
        self.gcn2 = GraphConvLayer(gcn_dim, gcn_dim, A_norm)

        od_feat_dim = 1 + 2 * gcn_dim
        self.od_proj = nn.Linear(od_feat_dim, gru_hidden)
        self.gru = nn.GRU(
            gru_hidden,
            gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.time_proj = nn.Linear(time_feat_dim, 16)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        o_idx = torch.tensor([self.node2idx[o] for o, d in od_pairs], dtype=torch.long)
        d_idx = torch.tensor([self.node2idx[d] for o, d in od_pairs], dtype=torch.long)
        self.register_buffer("o_idx", o_idx)
        self.register_buffer("d_idx", d_idx)

    def forward(self, x_flow: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        B, L, N_OD = x_flow.shape
        device = x_flow.device

        node_ids = torch.arange(self.n_node, device=device)
        emb = self.node_emb(node_ids).unsqueeze(0).expand(B, -1, -1)
        g = self.gcn2(self.gcn1(emb))
        o_feat = g[:, self.o_idx, :]
        d_feat = g[:, self.d_idx, :]

        od_feats = []
        for t in range(L):
            flow_t = x_flow[:, t, :].unsqueeze(-1)
            feat_t = torch.cat([flow_t, o_feat, d_feat], dim=-1)
            od_feats.append(F.relu(self.od_proj(feat_t)))
        od_seq = torch.stack(od_feats, dim=2)

        flat = od_seq.reshape(B * N_OD, L, -1)
        gru_out, _ = self.gru(flat)
        h_last = gru_out[:, -1, :].view(B, N_OD, -1)

        t_feat = F.relu(self.time_proj(x_time[:, -1, :]))
        t_feat = t_feat.unsqueeze(1).expand(-1, N_OD, -1)
        combined = torch.cat([h_last, t_feat], dim=-1)
        out = self.fc(combined).squeeze(-1)
        return torch.clamp(out, 0.0, 1.0)


class PureGRUModel(nn.Module):
    def __init__(
        self,
        n_od: int,
        gru_hidden: int,
        time_feat_dim: int,
        seq_len: int,
    ):
        super().__init__()
        self.n_od = n_od
        self.seq_len = seq_len
        self.gru = nn.GRU(
            1,
            gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.time_proj = nn.Linear(time_feat_dim, 16)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_flow: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        B, L, N_OD = x_flow.shape
        x_flat = x_flow.permute(0, 2, 1).reshape(B * N_OD, L, 1)
        gru_out, _ = self.gru(x_flat)
        h_last = gru_out[:, -1, :].view(B, N_OD, -1)
        t_feat = F.relu(self.time_proj(x_time[:, -1, :]))
        t_feat = t_feat.unsqueeze(1).expand(-1, N_OD, -1)
        combined = torch.cat([h_last, t_feat], dim=-1)
        out = self.fc(combined).squeeze(-1)
        return torch.clamp(out, 0.0, 1.0)
