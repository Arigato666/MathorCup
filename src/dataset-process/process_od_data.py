"""
OD 客流数据处理流水线（与赛题方案一致）：
1. 剔除自环 OD
2. 15 分钟时间片聚合
3. 全时空笛卡尔积 + 零填充
4. 分层 IQR（仅非零）异常检测 + 同星期几同小时历史中位数替换
5. 前 21 天训练 / 后 7 天测试划分
6. 导出 TII / WDI / PVR 指标表

用法（在仓库根目录或本目录）：
  python process_od_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-5

# 路径：数据与输出相对本脚本所在目录
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "dataset"
OUT_DIR = HERE / "output"


def get_period(hour: int, minute: int) -> str:
    t = hour + minute / 60.0
    if 7 <= t < 9:
        return "AM_Peak"
    if 17 <= t < 19:
        return "PM_Peak"
    if 9 <= t < 17:
        return "Off_Peak"
    if t >= 22 or t < 6:
        return "Night"
    return "Shoulder"


def load_and_aggregate(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["time_slot"] = df_raw["timestamp"].dt.floor("15min")

    df_nonself = df_raw[df_raw["in_station"] != df_raw["out_station"]].copy()
    df_agg = (
        df_nonself.groupby(["time_slot", "in_station", "out_station"], as_index=False)["flow"]
        .sum()
    )
    return df_agg


def cartesian_complete(df_agg: pd.DataFrame, df_raw_times: pd.DataFrame) -> pd.DataFrame:
    nodes = [f"Node_{i}" for i in range(8)]
    od_pairs = [(o, d) for o in nodes for d in nodes if o != d]

    ts = pd.to_datetime(df_raw_times["timestamp"])
    start_time = ts.min().normalize()
    end_time = ts.max().normalize() + pd.Timedelta(days=1)
    all_slots = pd.date_range(start=start_time, end=end_time, freq="15min", inclusive="left")

    full_index = [(t, o, d) for t in all_slots for o, d in od_pairs]
    df_full = pd.DataFrame(full_index, columns=["time_slot", "in_station", "out_station"])
    df_full = df_full.merge(
        df_agg, on=["time_slot", "in_station", "out_station"], how="left"
    )
    df_full["flow"] = df_full["flow"].fillna(0).astype(float)

    df_full["hour"] = df_full["time_slot"].dt.hour
    df_full["minute"] = df_full["time_slot"].dt.minute
    df_full["weekday"] = df_full["time_slot"].dt.weekday
    df_full["is_weekend"] = df_full["weekday"] >= 5
    df_full["day_index"] = (df_full["time_slot"].dt.normalize() - start_time).dt.days

    return df_full, start_time


def layered_iqr_repair(df_full: pd.DataFrame) -> pd.DataFrame:
    df = df_full.copy()
    df["is_outlier"] = False
    df["flow_clean"] = df["flow"].copy()

    for (o, d), grp_idx in df.groupby(["in_station", "out_station"]).groups.items():
        group = df.loc[grp_idx]
        non_zero_mask = group["flow"] > 0
        non_zero_vals = group.loc[non_zero_mask, "flow"]

        if len(non_zero_vals) < 10:
            continue

        q1 = non_zero_vals.quantile(0.25)
        q3 = non_zero_vals.quantile(0.75)
        upper_bound = q3 + 3 * (q3 - q1)

        outlier_idx = group[(group["flow"] > upper_bound) & non_zero_mask].index
        df.loc[outlier_idx, "is_outlier"] = True

        for idx in outlier_idx:
            row = df.loc[idx]
            ctx = (
                (df["in_station"] == o)
                & (df["out_station"] == d)
                & (~df["is_outlier"])
                & (df["hour"] == row["hour"])
                & (df["weekday"] == row["weekday"])
                & (df["flow"] > 0)
            )
            ctx_vals = df.loc[ctx, "flow"]
            df.at[idx, "flow_clean"] = (
                float(ctx_vals.median()) if len(ctx_vals) > 0 else float(non_zero_vals.median())
            )

    df["flow"] = df["flow_clean"]
    return df


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["period"] = df.apply(lambda r: get_period(int(r["hour"]), int(r["minute"])), axis=1)

    peak_means = (
        df[~df["is_weekend"]]
        .groupby(["in_station", "out_station", "period"])["flow"]
        .mean()
        .unstack("period")
        .fillna(0)
    )
    am = peak_means.get("AM_Peak", pd.Series(0, index=peak_means.index))
    pm = peak_means.get("PM_Peak", pd.Series(0, index=peak_means.index))
    tii_df = pd.DataFrame(
        {"AM_mean": am, "PM_mean": pm, "TII": (am - pm) / (am + pm + EPS)}
    ).reset_index()

    day_means = (
        df.groupby(["in_station", "out_station", "is_weekend"])["flow"]
        .mean()
        .unstack("is_weekend")
        .fillna(0)
    )
    day_means = day_means.rename(columns={False: "weekday_mean", True: "weekend_mean"})
    wdi_df = day_means.copy()
    wdi_df["WDI"] = (wdi_df["weekday_mean"] - wdi_df["weekend_mean"]) / (
        wdi_df["weekday_mean"] + wdi_df["weekend_mean"] + EPS
    )
    wdi_df = wdi_df.reset_index()

    pvr_base = (
        df[~df["is_weekend"]]
        .groupby(["in_station", "out_station", "period"])["flow"]
        .mean()
        .unstack("period")
        .fillna(0)
    )
    peak_mean = pvr_base[["AM_Peak", "PM_Peak"]].max(axis=1) if not pvr_base.empty else pd.Series()
    offpeak = pvr_base.get("Off_Peak", pd.Series(EPS, index=pvr_base.index))
    pvr_df = pd.DataFrame({"peak_mean": peak_mean, "offpeak_mean": offpeak})
    pvr_df["PVR"] = pvr_df["peak_mean"] / (pvr_df["offpeak_mean"] + EPS)
    pvr_df = pvr_df.reset_index()

    return tii_df, wdi_df, pvr_df


def main() -> int:
    csv_path = DATA_DIR / "toy_od_flow_data.csv"
    if not csv_path.exists():
        print(f"未找到数据文件: {csv_path}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    print(f"原始数据行数: {len(df_raw)}")
    print(f"时间范围: {df_raw['timestamp'].min()} ~ {df_raw['timestamp'].max()}")

    df_agg = load_and_aggregate(df_raw)
    n_self = len(df_raw) - len(df_raw[df_raw["in_station"] != df_raw["out_station"]])
    print(f"去除自环后: {len(df_agg)} 组聚合键对应行（移除自环记录约 {n_self} 条）")

    df_full, start_time = cartesian_complete(df_agg, df_raw)
    sparsity = (df_full["flow"] == 0).mean()
    print(f"补全后: {len(df_full)} 行 | 稀疏度(零占比): {sparsity:.1%}")

    df_full = layered_iqr_repair(df_full)
    outlier_cnt = int(df_full["is_outlier"].sum())
    nz = (df_full["flow"] > 0).sum()
    # 注：is_outlier 标记基于原始 flow；修复后 flow 已替换
    print(
        f"检测到异常值: {outlier_cnt} 条"
        + (f"（占非零时间片 {outlier_cnt / max(nz, 1):.2%}）" if nz else "")
    )

    df_train = df_full[df_full["day_index"] < 21].copy()
    df_test = df_full[df_full["day_index"] >= 21].copy()
    print(
        f"训练集: {df_train['time_slot'].nunique()} 个时间片 | "
        f"测试集: {df_test['time_slot'].nunique()} 个时间片"
    )

    cols_export = [
        "time_slot",
        "in_station",
        "out_station",
        "flow",
        "hour",
        "minute",
        "weekday",
        "is_weekend",
        "day_index",
    ]
    df_full[cols_export].to_csv(OUT_DIR / "od_cleaned_full.csv", index=False)
    df_train[cols_export].to_csv(OUT_DIR / "od_train.csv", index=False)
    df_test[cols_export].to_csv(OUT_DIR / "od_test.csv", index=False)

    tii_df, wdi_df, pvr_df = compute_metrics(df_full)
    metrics = tii_df.merge(wdi_df, on=["in_station", "out_station"], how="outer")
    metrics = metrics.merge(
        pvr_df[["in_station", "out_station", "PVR", "peak_mean", "offpeak_mean"]],
        on=["in_station", "out_station"],
        how="outer",
    )
    metrics.to_csv(OUT_DIR / "od_metrics_summary.csv", index=False)

    print("\n=== 完成 ===")
    print(f"输出目录: {OUT_DIR}")
    print("  od_cleaned_full.csv, od_train.csv, od_test.csv, od_metrics_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
