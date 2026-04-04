# Problem2 消融实验说明

在 `src/problem2` 目录下执行（确保已安装项目根目录 `requirements.txt` 依赖）。

## 有图 vs 无图（邻接矩阵）

两次运行除 `USE_GRAPH` 外保持 [`option.py`](option.py) 中其它超参一致（含 `SEED`），对比测试集 RMSE / MAE / R² / MAPE / ACC。

**引入物理拓扑 + 多跳邻接：** 设置 `USE_GRAPH = True`，在 `src/problem2` 下执行：

```bash
cd src/problem2
python main.py
```

**纯时序基线（`A = I`，不使用邻接）：** 将 `USE_GRAPH = False` 后再次运行同一命令。

记录两次运行结束时打印的「测试集（第四周）」指标与学习曲线文件名 `learning_curve_{model}_{G|NOG}.png`。

## 可选：多跳阶数 `k_hops`

在 `option.py` 中修改 `k_hops`（例如 `1`、`2`、`4`），保持 `USE_GRAPH=True`，观察验证集与测试集指标变化；`k_hops=1` 时仅使用原始邻接矩阵的一跳边（循环内不叠加更高次幂）。

## 可选：log1p

`USE_LOG1P = True`（默认）对稀疏客流更友好；设为 `False` 可对照旧版「仅 MinMax」管线。
