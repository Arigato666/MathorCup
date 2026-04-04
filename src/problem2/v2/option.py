"""v2: 改进版配置（相对本文件目录，数据沿用上级 problem2/data）。"""
import torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT.parent / "data"

TRAIN_FILE = str(_DATA / "od_train.csv")
TEST_FILE = str(_DATA / "od_test.csv")
ADJ_FILE = str(_DATA / "toy_network_adjacency.csv")

SEQ_LEN = 12
HIDDEN_DIM = 96
EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
k_hops = 2

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_GRAPH = True
MODEL = "STGNN"  # "STGNN" | "STGAT"

# 导出第四周预测（与测试标签对齐的滑窗样本；提交时可再按需展开为全时间片）
EXPORT_PRED_CSV = str(_ROOT / "predictions_week4_aligned.csv")
