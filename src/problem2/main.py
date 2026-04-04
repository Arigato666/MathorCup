import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import option
from dataset import prep_data, get_adj
from model import STGNN, STGAT

warnings.filterwarnings('ignore')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate_epoch(model, x, y, scaler, n_nodes):
    model.eval()
    with torch.no_grad():
        preds = model(x).cpu().numpy()
        trues = y.cpu().numpy()

    # 反归一化
    preds_inv = scaler.inverse_transform(preds.reshape(-1, n_nodes ** 2))
    trues_inv = scaler.inverse_transform(trues.reshape(-1, n_nodes ** 2))

    # 截断负数客流
    preds_inv = np.maximum(preds_inv, 0)

    # 计算三大指标
    rmse = np.sqrt(mean_squared_error(trues_inv, preds_inv))
    mae = mean_absolute_error(trues_inv, preds_inv)
    r2 = r2_score(trues_inv, preds_inv)

    return rmse, mae, r2


def main():
    set_seed(option.SEED)
    print(f"Using device: {option.DEVICE}")

    # 1. 准备数据
    x_tr, y_tr, x_te, y_te, scaler, n_nodes = prep_data(
        option.TRAIN_FILE, option.TEST_FILE, option.SEQ_LEN
    )
    A_norm = get_adj(option.ADJ_FILE, n_nodes)

    x_tr, y_tr = x_tr.to(option.DEVICE), y_tr.to(option.DEVICE)
    x_te, y_te = x_te.to(option.DEVICE), y_te.to(option.DEVICE)
    A_norm = A_norm.to(option.DEVICE)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=option.BATCH_SIZE, shuffle=True)

    # 2. 实例化模型与优化器
    if(option.model=='STGNN'):
        model = STGNN(n_nodes, option.HIDDEN_DIM, A_norm).to(option.DEVICE)
    elif(option.model=='STGAT'):
        model=STGAT(n_nodes, option.HIDDEN_DIM, A_norm).to(option.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=option.LR)
    loss_fn = nn.MSELoss()

    # 记录原始指标
    test_rmse_history = []
    test_mae_history = []
    test_r2_history = []

    # 记录三者平均后的综合得分
    train_score_history = []
    test_score_history = []

    # 3. 训练循环
    print("\n[Start Training]")
    for ep in range(option.EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {ep + 1}/{option.EPOCHS}', leave=False)
        for batch_x, batch_y in pbar:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 每一轮结束评估
        rmse_tr, mae_tr, r2_tr = evaluate_epoch(model, x_tr, y_tr, scaler, n_nodes)
        rmse_te, mae_te, r2_te = evaluate_epoch(model, x_te, y_te, scaler, n_nodes)

        # 统一步调：把 R2 变成越小越好 (1 - R2)，然后求平均
        score_tr = (rmse_tr + mae_tr + (1 - r2_tr)) / 3.0
        score_te = (rmse_te + mae_te + (1 - r2_te)) / 3.0

        test_rmse_history.append(rmse_te)
        test_mae_history.append(mae_te)
        test_r2_history.append(r2_te)

        train_score_history.append(score_tr)
        test_score_history.append(score_te)

        if (ep + 1) % 5 == 0:
            print(f"Epoch [{ep + 1}/{option.EPOCHS}] | Train Score: {score_tr:.4f} | Test Score: {score_te:.4f}")

    # 4. === 打印最后一轮的三大指标 ===
    final_rmse = test_rmse_history[-1]
    final_mae = test_mae_history[-1]
    final_r2 = test_r2_history[-1]

    print("\n" + "=" * 45)
    print(f"🏁 【最终模型全局成绩单 (Epoch {option.EPOCHS} 结束时)】")
    print(f"   是否引入图结构 (USE_GRAPH) : {option.USE_GRAPH}")
    print("-" * 45)
    print(f"   均方根误差 (RMSE) : {final_rmse:.4f}")
    print(f"   平均绝对误差 (MAE) : {final_mae:.4f}")
    print(f"   决定系数 (R²)     : {final_r2:.4f}")
    print("=" * 45)

    # 5. 绘制综合得分 (Combined Score) 学习曲线
    best_epoch = np.argmin(test_score_history)
    best_score = test_score_history[best_epoch]

    plt.figure(figsize=(10, 6))
    epochs_range = range(1, option.EPOCHS + 1)

    plt.plot(epochs_range, train_score_history, label='Train Combined Score', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, test_score_history, label='Test Combined Score', color='#ff7f0e', linewidth=2)

    # 圈出综合得分的最优拐点（极小值）
    plt.scatter(best_epoch + 1, best_score, color='red', s=100, zorder=5)
    plt.annotate(f'Lowest Combined Score:\nEpoch: {best_epoch + 1}\nScore: {best_score:.4f}',
                 xy=(best_epoch + 1, best_score),
                 xytext=(best_epoch + 1 + 2, best_score + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    title_suffix = "with Graph" if option.USE_GRAPH else "without Graph (Baseline)"
    plt.title(f'Learning Curve: Average Combined Score ({title_suffix})', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Combined Score = (RMSE + MAE + (1-R²)) / 3', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = f"learning_curve_avg_{'STGNN' if option.USE_GRAPH else 'Baseline'}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n折线图已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()