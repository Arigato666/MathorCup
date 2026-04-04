import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import option
from dataset import prep_data
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(trues, preds):
    preds = np.maximum(preds, 0)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    return rmse, mae, r2


def main():
    print("Loading and preparing data...")
    # 获取测试集真值和 Scaler
    x_tr, y_tr, x_te, y_te, scaler, n_nodes = prep_data(
        option.TRAIN_FILE, option.TEST_FILE, option.SEQ_LEN
    )

    # =============== 修复维度匹配问题 ===============
    # 直接使用 (-1, n_nodes**2) 来适配独立归一化的 Scaler
    trues_scaled = y_te.numpy()
    trues_inv = scaler.inverse_transform(trues_scaled.reshape(-1, n_nodes ** 2))

    y_tr_scaled = y_tr.numpy()
    y_tr_inv = scaler.inverse_transform(y_tr_scaled.reshape(-1, n_nodes ** 2))
    # ================================================

    # 🤡 Baseline A: 纯瞎猜 (Random Guess)
    # 按照每个 OD 对的历史最大值进行独立的随机猜测
    max_flow_per_od = np.max(y_tr_inv, axis=0)  # 获取64个OD的独立最大值
    np.random.seed(42)

    # 对每个 OD 生成独立的随机预测
    preds_random = np.zeros_like(trues_inv)
    for i in range(n_nodes ** 2):
        preds_random[:, i] = np.random.uniform(0, max_flow_per_od[i], size=trues_inv.shape[0])

    rmse_rand, mae_rand, r2_rand = calculate_metrics(trues_inv, preds_random)

    # 🦥 Baseline B: 躺平策略 / 历史均值 (Historical Average)
    mean_flow_per_od = np.mean(y_tr_inv, axis=0)
    preds_mean = np.tile(mean_flow_per_od, (trues_inv.shape[0], 1))

    rmse_mean, mae_mean, r2_mean = calculate_metrics(trues_inv, preds_mean)

    print("\n" + "=" * 55)
    print("🎲 【纯瞎猜策略 (Random Guess) 的成绩单】")
    print("-" * 55)
    print(f"   RMSE : {rmse_rand:.4f}")
    print(f"   MAE  : {mae_rand:.4f}")
    print(f"   R²   : {r2_rand:.4f}")

    print("\n" + "=" * 55)
    print("🦥 【躺平策略 (Historical Average) 的成绩单】")
    print("-" * 55)
    print(f"   RMSE : {rmse_mean:.4f}")
    print(f"   MAE  : {mae_mean:.4f}")
    print(f"   R²   : {r2_mean:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()