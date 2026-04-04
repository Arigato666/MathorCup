import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import option
from dataset import prep_data, get_adj
from model import STGNN, STGAT
from metrics import evaluate_metrics, combined_score

warnings.filterwarnings("ignore")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _run_name():
    g = "G" if option.USE_GRAPH else "NOG"
    return f"{option.model}_{g}"


def _fmt_acc_rel(acc_rel):
    """打印多档相对误差命中率：|ŷ-y|/y < τ 的样本占比（非分类准确率）。"""
    parts = []
    for t in sorted(acc_rel.keys()):
        pct = int(round(t * 100))
        parts.append(f"Hit@τ={pct}%:{acc_rel[t]:.3f}")
    return " ".join(parts)


def main():
    set_seed(option.SEED)
    print(f"Using device: {option.DEVICE}")
    print(f"Experiment: model={option.model}, USE_GRAPH={option.USE_GRAPH}, k_hops={option.k_hops}, USE_LOG1P={option.USE_LOG1P}")

    x_train, y_train, x_val, y_val, x_test, y_test, scaler, n_nodes, meta = prep_data(
        option.TRAIN_FILE, option.TEST_FILE, option.SEQ_LEN
    )
    use_log1p = meta["use_log1p"]
    flow_bin_edges = meta.get("flow_bin_edges")

    A_norm = get_adj(option.ADJ_FILE, n_nodes)

    x_train = x_train.to(option.DEVICE)
    y_train = y_train.to(option.DEVICE)
    x_val = x_val.to(option.DEVICE)
    y_val = y_val.to(option.DEVICE)
    x_test = x_test.to(option.DEVICE)
    y_test = y_test.to(option.DEVICE)
    A_norm = A_norm.to(option.DEVICE)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=option.BATCH_SIZE,
        shuffle=True,
    )

    if option.model == "STGNN":
        model = STGNN(n_nodes, option.HIDDEN_DIM, A_norm).to(option.DEVICE)
    else:
        model = STGAT(n_nodes, option.HIDDEN_DIM, A_norm).to(option.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=option.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()

    val_rmse_history = []
    val_mae_history = []
    val_r2_history = []
    val_mape_history = []
    val_acc_history = []
    train_score_history = []
    val_score_history = []

    best_val = float("inf")
    best_state = None
    patience_left = (
        option.EARLY_STOP_PATIENCE
        if option.EARLY_STOP_PATIENCE > 0
        else float("inf")
    )

    print("\n[Start Training]")
    stopped_epoch = option.EPOCHS

    for ep in range(option.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep + 1}/{option.EPOCHS}", leave=False)
        for batch_x, batch_y in pbar:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        m_tr = evaluate_metrics(
            model,
            x_train,
            y_train,
            scaler,
            n_nodes,
            use_log1p,
            option.FLOW_EPS,
            option.ACC_REL_TAUS,
            option.DEVICE,
            flow_bin_edges,
        )
        m_val = evaluate_metrics(
            model,
            x_val,
            y_val,
            scaler,
            n_nodes,
            use_log1p,
            option.FLOW_EPS,
            option.ACC_REL_TAUS,
            option.DEVICE,
            flow_bin_edges,
        )

        score_tr = combined_score(m_tr)
        score_val = combined_score(m_val)
        scheduler.step(score_val)

        val_rmse_history.append(m_val["rmse"])
        val_mae_history.append(m_val["mae"])
        val_r2_history.append(m_val["r2"])
        val_mape_history.append(m_val["mape"])
        val_acc_history.append(m_val["acc_rel"].get(float(option.ACC_REL_TAUS[0]), float("nan")))
        train_score_history.append(score_tr)
        val_score_history.append(score_val)

        if score_val < best_val:
            best_val = score_val
            best_state = copy.deepcopy(model.state_dict())
            if option.EARLY_STOP_PATIENCE > 0:
                patience_left = option.EARLY_STOP_PATIENCE
        else:
            if option.EARLY_STOP_PATIENCE > 0:
                patience_left -= 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"Epoch [{ep + 1}/{option.EPOCHS}] | "
                f"train_score={score_tr:.4f} val_score={score_val:.4f} | "
                f"val RMSE={m_val['rmse']:.4f} MAE={m_val['mae']:.4f} R²={m_val['r2']:.4f} | "
                f"MAPE={m_val['mape']:.4f} | {_fmt_acc_rel(m_val['acc_rel'])}"
            )

        if option.EARLY_STOP_PATIENCE > 0 and patience_left <= 0:
            stopped_epoch = ep + 1
            print(f"[Early stopping] at epoch {stopped_epoch}, best val_score={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    m_test = evaluate_metrics(
        model,
        x_test,
        y_test,
        scaler,
        n_nodes,
        use_log1p,
        option.FLOW_EPS,
        option.ACC_REL_TAUS,
        option.DEVICE,
        flow_bin_edges,
    )

    print("\n" + "=" * 52)
    print(f"【测试集（第四周）】model={option.model} USE_GRAPH={option.USE_GRAPH}")
    print("-" * 52)
    print(f"  RMSE : {m_test['rmse']:.4f}")
    print(f"  MAE  : {m_test['mae']:.4f}")
    print(f"  R²   : {m_test['r2']:.4f}")
    print(f"  MAPE : {m_test['mape']:.4f}  (y>{option.FLOW_EPS})")
    print(f"  WAPE : {m_test['wape']:.4f}")
    print("  相对误差命中率：在 y>eps 格点上 |ŷ-y|/y < τ")
    print(f"  {_fmt_acc_rel(m_test['acc_rel'])}")
    print(
        f"  整数一致率(仅y>eps): {m_test['acc_round_nz']:.4f}  "
        f"(round(ŷ)=round(y)，回归任务里通常低于分类准确率)"
    )
    print(
        f"  分档准确率(训练集分位分{option.ACC_BIN_K}类): {m_test['acc_bin']:.4f}  "
        f"(将 y 与 ŷ 按同一箱边界分类后，类别完全一致的比例)"
    )
    print("=" * 52)

    best_epoch = int(np.argmin(val_score_history)) + 1
    best_val_sc = float(np.min(val_score_history))

    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_score_history) + 1)
    plt.plot(epochs_range, train_score_history, label="Train combined score", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, val_score_history, label="Val combined score", color="#ff7f0e", linewidth=2)
    plt.scatter(best_epoch, best_val_sc, color="red", s=100, zorder=5)
    plt.annotate(
        f"Best val:\nepoch {best_epoch}\nscore {best_val_sc:.4f}",
        xy=(best_epoch, best_val_sc),
        xytext=(best_epoch + 1, best_val_sc + 0.05),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )
    title = f"{_run_name()} | log1p={use_log1p}"
    plt.title(f"Learning curve ({title})", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("(RMSE + MAE + (1-R²)) / 3")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    save_path = f"learning_curve_{_run_name()}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n学习曲线已保存: {save_path}")


if __name__ == "__main__":
    main()
