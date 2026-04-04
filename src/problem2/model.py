import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class STGAT(nn.Module):
    """
    时空图注意力网络 (STGAT + GRU)
    完美响应赛题中“沿物理拓扑扩散”与“应对高波动稀疏性”的要求
    """

    def __init__(self, n_nodes, h_dim, A):
        super().__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim

        # 将传入的归一化矩阵恢复成 0-1 掩码矩阵 (用于限制只在物理相连的边上算注意力)
        self.A_mask = (A > 0).float()

        # 空间特征提取: GAT 所需的线性变换参数
        self.W = nn.Linear(n_nodes, h_dim)

        # =============== 修复部分 ===============
        # 直接定义为 2D 张量，完美支持 Xavier 初始化
        self.a_src = nn.Parameter(torch.empty(h_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(h_dim, 1))
        nn.init.xavier_normal_(self.a_src)
        nn.init.xavier_normal_(self.a_dst)
        # ========================================

        self.leakyrelu = nn.LeakyReLU(0.2)

        # 时间特征提取: GRU
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, n_nodes)

    def forward(self, x):
        b, s, n, _ = x.shape

        # 1. 图注意力空间扩散 (GAT)
        h = self.W(x)  # (b, s, n, h_dim)

        # 计算注意力得分 (利用 matmul 的自动广播机制)
        # h: (b, s, n, h_dim) 与 a: (h_dim, 1) 相乘 -> (b, s, n, 1)，squeeze后变 (b, s, n)
        attn_src = torch.matmul(h, self.a_src).squeeze(-1)
        attn_dst = torch.matmul(h, self.a_dst).squeeze(-1)

        # e: (b, s, n, n)
        e = attn_src.unsqueeze(3) + attn_dst.unsqueeze(2)
        e = self.leakyrelu(e)

        # Mask 机制：只允许沿着“物理连接的网络边”扩散
        zero_vec = -9e15 * torch.ones_like(e)
        mask = self.A_mask.unsqueeze(0).unsqueeze(0).expand(b, s, n, n).to(e.device)
        attention = torch.where(mask > 0, e, zero_vec)

        # Softmax 归一化
        attention = F.softmax(attention, dim=-1)

        # 聚合邻居特征
        h_sp = torch.matmul(attention, h)
        h_sp = F.elu(h_sp)

        # 残差连接
        h_sp = h_sp + h

        # 2. 时序演化 (GRU)
        h_sp = h_sp.permute(0, 2, 1, 3).reshape(b * n, s, self.h_dim)
        _, ht = self.gru(h_sp)

        out = self.fc(ht[-1])
        return out.reshape(b, n, n)

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