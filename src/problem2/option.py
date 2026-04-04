import torch

# ================= 数据路径 =================
TRAIN_FILE = "data/od_train.csv"
TEST_FILE = "data/od_test.csv"
ADJ_FILE = "data/toy_network_adjacency.csv"

# ================= 超参数 (Hyperparameters) =================
SEQ_LEN = 6          # 历史时间步窗口长度 (看过去多少步)
HIDDEN_DIM = 64      # GCN 和 GRU 的隐藏层维度
EPOCHS = 26          # 训练轮次上限
BATCH_SIZE = 32      # 批大小
LR = 0.005           # 学习率
k_hops = 2           # 邻接多跳上限（1 表示仅物理 1 跳，>=2 时累加 A^2…）

# 前 21 天中末尾若干天作为验证集（标签落在该时段）
VAL_DAYS = 3
# 对客流先 log1p 再 MinMax（稀疏 OD 常用）；False 则与旧管线一致
USE_LOG1P = True

# 早停：验证集 combined score 连续 patience 轮不降则停止；0 表示关闭
EARLY_STOP_PATIENCE = 6
# 相对误差「命中率」：仅在 y_true > FLOW_EPS 的 OD 格点上，统计 |ŷ-y|/y < τ 的比例。
# 注意：这是回归任务上的相对误差命中比例，不是分类问题里的「准确率」；τ 越小越苛刻。
FLOW_EPS = 1e-3
ACC_REL_TAUS = (0.2, 0.5, 1.0)  # 多档 τ；例如 0.2 表示误差需小于真实值的 20%

# 分档「准确率」：用训练集正客流的分位数作为箱边界，将 y 与 ŷ 分到同一组 K 类后算分类准确率
ACC_BIN_K = 5

# ================= 实验控制 =================
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 【消融实验】False 时 get_adj 返回单位阵（无图）
USE_GRAPH = True
# 模型：'STGNN' 或 'STGAT'
model = "STGAT"