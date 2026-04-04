"""v4：加权 Huber、双子图学习曲线、更长训练；数据逻辑与 v3 一致（全时间轴）。"""
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT.parent / "data"

TRAIN_CSV = str(_DATA / "od_train.csv")
TEST_CSV = str(_DATA / "od_test.csv")
ADJ_CSV = str(_DATA / "toy_network_adjacency.csv")

SEQ_LEN = 12
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练损失：True 时用加权 Huber（大流量 OD 在归一化空间权重更高）
USE_WEIGHTED_HUBER = True
HUBER_DELTA = 1.0
WEIGHTED_HUBER_ALPHA = 2.0

NODE_EMB_DIM = 16
GCN_DIM = 32
GRU_HIDDEN = 64
TIME_FEAT_DIM = 5

OUT_SUBMISSION = str(_ROOT / "prediction_week4_v4.csv")
OUT_CURVE = str(_ROOT / "learning_curve_v4.png")
