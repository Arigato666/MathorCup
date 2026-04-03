import torch

# ================= 数据路径 =================
TRAIN_FILE = 'data/od_train.csv'
TEST_FILE = 'data/od_test.csv'
ADJ_FILE = 'data/toy_network_adjacency.csv'

# ================= 超参数 (Hyperparameters) =================
SEQ_LEN = 6          # 历史时间步窗口长度 (看过去多少步)
HIDDEN_DIM = 64      # GCN 和 GRU 的隐藏层维度
EPOCHS = 20         # 训练轮次
BATCH_SIZE = 32      # 批大小
LR = 0.005           # 学习率

# ================= 实验控制 =================
SEED = 42            # 随机种子，保证每次跑的分数一样
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 【消融实验开关】
USE_GRAPH = True