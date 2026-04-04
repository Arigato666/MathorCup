"""v3: ST-GCN + GRU 与纯 GRU 消融；数据路径指向 problem2/data。"""
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT.parent / "data"

TRAIN_CSV = str(_DATA / "od_train.csv")
TEST_CSV = str(_DATA / "od_test.csv")
ADJ_CSV = str(_DATA / "toy_network_adjacency.csv")

SEQ_LEN = 12
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NODE_EMB_DIM = 16
GCN_DIM = 32
GRU_HIDDEN = 64
TIME_FEAT_DIM = 5

OUT_SUBMISSION = str(_ROOT / "prediction_week4_v3.csv")
OUT_CURVE = str(_ROOT / "learning_curve_v3_stgcn.png")
