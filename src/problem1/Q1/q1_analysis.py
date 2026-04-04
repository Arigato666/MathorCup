"""
Problem 1 - Q1:
Quantify OD differences across weekday/weekend and AM/PM peaks.

Input:
  src/dataset-process/output/od_cleaned_full.csv

Output:
  src/problem1/output/
    - q1_metrics_all_od.csv
    - q1_top_tables.csv
    - figures/*.png
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
(HERE / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

EPS = 1e-5
NODES = [f"Node_{i}" for i in range(8)]
SRC = HERE.parent.parent  # repo src/: Q1 -> problem1 -> src
INPUT = SRC / "dataset-process" / "output" / "od_cleaned_full.csv"
OUT_DIR = HERE / "output"
FIG_DIR = OUT_DIR / "figures"
PRE_FIG_DIR = SRC / "dataset-process" / "output" / "figures"

PALETTE = {
    "weekday": "#355C7D",
    "weekend": "#C06C84",
    "up": "#6C5B7B",
    "down": "#F67280",
}


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


def compute_cc(df: pd.DataFrame, o: str, d: str, max_lag_hours: int = 12) -> tuple[float, float]:
    wd = df[~df["is_weekend"]]
    seq_od = (
        wd[(wd["in_station"] == o) & (wd["out_station"] == d)]
        .groupby(["hour", "minute"])["flow"]
        .mean()
    )
    seq_do = (
        wd[(wd["in_station"] == d) & (wd["out_station"] == o)]
        .groupby(["hour", "minute"])["flow"]
        .mean()
    )
    idx = seq_od.index.union(seq_do.index)
    seq_od = seq_od.reindex(idx, fill_value=0).sort_index().values.astype(float)
    seq_do = seq_do.reindex(idx, fill_value=0).sort_index().values.astype(float)
    if len(seq_od) < 10:
        return np.nan, np.nan

    n_od = (seq_od - seq_od.mean()) / (seq_od.std() + EPS)
    n_do = (seq_do - seq_do.mean()) / (seq_do.std() + EPS)
    corr = correlate(n_od, n_do, mode="full")
    lags = np.arange(-(len(n_od) - 1), len(n_od)) * 15
    mask = np.abs(lags) <= max_lag_hours * 60
    sub_corr = corr[mask] / len(n_od)
    sub_lags = lags[mask]
    best = int(np.argmax(sub_corr))
    return float(sub_corr[best]), float(sub_lags[best])


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = df.apply(lambda r: get_period(int(r["hour"]), int(r["minute"])), axis=1)

    # TII
    peak_means = (
        df[~df["is_weekend"]]
        .groupby(["in_station", "out_station", "period"])["flow"]
        .mean()
        .unstack("period")
        .fillna(0)
    )
    am = peak_means.get("AM_Peak", pd.Series(0, index=peak_means.index))
    pm = peak_means.get("PM_Peak", pd.Series(0, index=peak_means.index))
    tii = ((am - pm) / (am + pm + EPS)).rename("TII")

    # WDI
    dw = (
        df.groupby(["in_station", "out_station", "is_weekend"])["flow"]
        .mean()
        .unstack("is_weekend")
        .fillna(0)
        .rename(columns={False: "weekday_mean", True: "weekend_mean"})
    )
    wdi = ((dw["weekday_mean"] - dw["weekend_mean"]) / (dw["weekday_mean"] + dw["weekend_mean"] + EPS)).rename("WDI")

    # PVR
    pvr_base = (
        df[~df["is_weekend"]]
        .groupby(["in_station", "out_station", "period"])["flow"]
        .mean()
        .unstack("period")
        .fillna(0)
    )
    peak = pvr_base[["AM_Peak", "PM_Peak"]].max(axis=1)
    offpeak = pvr_base.get("Off_Peak", pd.Series(EPS, index=pvr_base.index))
    pvr = (peak / (offpeak + EPS)).rename("PVR")

    metrics = (
        pd.concat([dw, tii, wdi, pvr], axis=1)
        .reset_index()
        .sort_values(["in_station", "out_station"])
    )

    ccs = []
    for _, row in metrics.iterrows():
        cc, lag = compute_cc(df, row["in_station"], row["out_station"])
        ccs.append((cc, lag))
    metrics["CC_max"] = [x[0] for x in ccs]
    metrics["CC_lag_min"] = [x[1] for x in ccs]
    return metrics


def plot_tii_wdi_heatmaps(metrics: pd.DataFrame) -> None:
    m_tii = np.full((8, 8), np.nan)
    m_wdi = np.full((8, 8), np.nan)
    for _, r in metrics.iterrows():
        i = int(r["in_station"].split("_")[1])
        j = int(r["out_station"].split("_")[1])
        m_tii[i, j] = r["TII"]
        m_wdi[i, j] = r["WDI"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.heatmap(
        m_tii,
        ax=axes[0],
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        xticklabels=NODES,
        yticklabels=NODES,
        cbar_kws={"label": "TII"},
    )
    axes[0].set_title("TII Matrix (OD pairs)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Destination")
    axes[0].set_ylabel("Origin")

    sns.heatmap(
        m_wdi,
        ax=axes[1],
        cmap="Spectral",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0,
        xticklabels=NODES,
        yticklabels=NODES,
        cbar_kws={"label": "WDI"},
    )
    axes[1].set_title("WDI Matrix (OD pairs)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Destination")
    axes[1].set_ylabel("Origin")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_tii_wdi_heatmaps.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_tii_bar(metrics: pd.DataFrame) -> None:
    d = metrics.copy()
    d["OD"] = d["in_station"] + "->" + d["out_station"]
    d = d.assign(abs_tii=d["TII"].abs()).sort_values("abs_tii", ascending=False).head(24)
    d = d.sort_values("TII")
    colors = np.where(d["TII"] >= 0, PALETTE["up"], PALETTE["down"])

    plt.figure(figsize=(10.5, 8.2))
    plt.barh(d["OD"], d["TII"], color=colors, alpha=0.92)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("TII")
    plt.ylabel("OD Pair")
    plt.title("Top-24 OD by |TII| (Tidal Intensity)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7, axis="x")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_tii_bar.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_weekday_weekend_scatter(metrics: pd.DataFrame) -> None:
    x = metrics["weekday_mean"].values
    y = metrics["weekend_mean"].values
    lim = max(np.max(x), np.max(y)) * 1.08
    lim = max(lim, 1.0)

    plt.figure(figsize=(7.8, 6.8))
    sc = plt.scatter(
        x,
        y,
        c=metrics["WDI"],
        cmap="coolwarm",
        s=85,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.88,
    )
    plt.plot([0, lim], [0, lim], "--", color="gray", linewidth=1.2, label="y=x")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Weekday Avg Flow")
    plt.ylabel("Weekend Avg Flow")
    plt.title("Weekday vs Weekend OD Flow (color=WDI)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left")
    cbar = plt.colorbar(sc)
    cbar.set_label("WDI")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q1_weekday_weekend_scatter.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_typical_curves(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    # Four representative ODs: strongest tidal, strongest weekday, strongest weekend, strongest peak concentration
    p1 = metrics.iloc[metrics["TII"].abs().idxmax()][["in_station", "out_station"]].tolist()
    p2 = metrics.iloc[metrics["WDI"].idxmax()][["in_station", "out_station"]].tolist()
    p3 = metrics.iloc[metrics["WDI"].idxmin()][["in_station", "out_station"]].tolist()
    p4 = metrics.iloc[metrics["PVR"].idxmax()][["in_station", "out_station"]].tolist()
    pairs = [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    fig.suptitle("OD Flow Temporal Pattern Analysis", fontsize=13, fontweight="bold", y=0.98)

    for ax, (o, d) in zip(axes.flatten(), pairs):
        sub = df[(df["in_station"] == o) & (df["out_station"] == d)].copy()
        sub["t"] = sub["hour"] + sub["minute"] / 60.0
        t = np.sort(sub["t"].unique())
        wd_raw = sub[~sub["is_weekend"]].groupby("t")["flow"].mean().reindex(t, fill_value=0).values
        we_raw = sub[sub["is_weekend"]].groupby("t")["flow"].mean().reindex(t, fill_value=0).values

        # Keep refined curve for readability while matching preprocess style.
        t_dense = np.linspace(0, 24, 289)
        wd_dense = np.interp(t_dense, t, wd_raw, left=0, right=0)
        we_dense = np.interp(t_dense, t, we_raw, left=0, right=0)
        wd = gaussian_filter1d(wd_dense, sigma=0.6)
        we = gaussian_filter1d(we_dense, sigma=0.6)

        ax.axvspan(7, 9, alpha=0.28, color="#F5DEB3", zorder=0, label="AM Peak")
        ax.axvspan(17, 19, alpha=0.28, color="#D8BFD8", zorder=0, label="PM Peak")
        ax.plot(t_dense, wd, color="#2166AC", lw=2.0, label="Weekday", zorder=2)
        ax.plot(t_dense, we, color="#D6604D", lw=1.8, ls="--", label="Weekend", zorder=2)

        pk = int(np.argmax(wd))
        ax.scatter(
            t_dense[pk],
            wd[pk],
            color="#2166AC",
            s=28,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.grid(alpha=0.28)
        rr = metrics[(metrics["in_station"] == o) & (metrics["out_station"] == d)].iloc[0]
        ax.set_title(
            f"{o} -> {d}    TII={rr['TII']:.3f}    WDI={rr['WDI']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Flow")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG_DIR / "q1_typical_curves.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_typical_curves_2up(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """
    Paper-friendly version: two figures, each containing 2 subplots (1x2).
    This avoids tiny subplots in the main text.
    """
    p1 = metrics.iloc[metrics["TII"].abs().idxmax()][["in_station", "out_station"]].tolist()
    p2 = metrics.iloc[metrics["WDI"].idxmax()][["in_station", "out_station"]].tolist()
    p3 = metrics.iloc[metrics["WDI"].idxmin()][["in_station", "out_station"]].tolist()
    p4 = metrics.iloc[metrics["PVR"].idxmax()][["in_station", "out_station"]].tolist()
    pairs = [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]

    def _plot_pair(ax, o: str, d: str) -> None:
        sub = df[(df["in_station"] == o) & (df["out_station"] == d)].copy()
        sub["t"] = sub["hour"] + sub["minute"] / 60.0
        t = np.sort(sub["t"].unique())
        wd_raw = sub[~sub["is_weekend"]].groupby("t")["flow"].mean().reindex(t, fill_value=0).values
        we_raw = sub[sub["is_weekend"]].groupby("t")["flow"].mean().reindex(t, fill_value=0).values

        t_dense = np.linspace(0, 24, 289)
        wd_dense = np.interp(t_dense, t, wd_raw, left=0, right=0)
        we_dense = np.interp(t_dense, t, we_raw, left=0, right=0)
        wd = gaussian_filter1d(wd_dense, sigma=0.6)
        we = gaussian_filter1d(we_dense, sigma=0.6)

        ax.axvspan(7, 9, alpha=0.28, color="#F5DEB3", zorder=0, label="AM Peak")
        ax.axvspan(17, 19, alpha=0.28, color="#D8BFD8", zorder=0, label="PM Peak")
        ax.plot(t_dense, wd, color="#2166AC", lw=2.0, label="Weekday", zorder=2)
        ax.plot(t_dense, we, color="#D6604D", lw=1.8, ls="--", label="Weekend", zorder=2)

        pk = int(np.argmax(wd))
        ax.scatter(
            t_dense[pk],
            wd[pk],
            color="#2166AC",
            s=28,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

        rr = metrics[(metrics["in_station"] == o) & (metrics["out_station"] == d)].iloc[0]
        ax.set_title(f"{o} -> {d}    TII={rr['TII']:.3f}    WDI={rr['WDI']:.3f}", fontsize=10)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Flow")
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.grid(alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)

    for tag, pair_list in [("a", pairs[:2]), ("b", pairs[2:])]:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        fig.suptitle("OD Flow Temporal Pattern Analysis", fontsize=13, fontweight="bold", y=1.02)
        _plot_pair(axes[0], *pair_list[0])
        _plot_pair(axes[1], *pair_list[1])
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"q1_typical_curves_2up_{tag}.png", bbox_inches="tight", dpi=300)
        plt.close()


def copy_preprocess_figures() -> None:
    """
    Optional reuse of figures already produced during dataset processing.
    Keeps one source-of-truth for common EDA visuals.
    """
    mapping = {
        "fig1_weekday_weekend_od_curves.png": "q1_from_preprocess_fig1_curves.png",
        "fig1_weekday_weekend_od_curves_1x2.png": "q1_from_preprocess_fig1_curves_1x2.png",
        "fig3_heatmap_hour_weekday.png": "q1_from_preprocess_fig3_heatmap.png",
    }
    for src_name, dst_name in mapping.items():
        src = PRE_FIG_DIR / src_name
        dst = FIG_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    df = pd.read_csv(INPUT, parse_dates=["time_slot"])
    metrics = build_metrics(df)
    metrics.to_csv(OUT_DIR / "q1_metrics_all_od.csv", index=False)

    top_tables = pd.concat(
        [
            metrics.assign(rank_type="|TII| top10", rank_score=metrics["TII"].abs()).nlargest(10, "rank_score"),
            metrics.assign(rank_type="WDI top10", rank_score=metrics["WDI"]).nlargest(10, "rank_score"),
            metrics.assign(rank_type="PVR top10", rank_score=metrics["PVR"]).nlargest(10, "rank_score"),
            metrics.assign(rank_type="CC top10", rank_score=metrics["CC_max"]).nlargest(10, "rank_score"),
        ],
        ignore_index=True,
    )
    top_tables.to_csv(OUT_DIR / "q1_top_tables.csv", index=False)

    plot_tii_wdi_heatmaps(metrics)
    plot_tii_bar(metrics)
    plot_weekday_weekend_scatter(metrics)
    plot_typical_curves(df, metrics)
    plot_typical_curves_2up(df, metrics)
    copy_preprocess_figures()

    print("Q1 done.")
    print(f"Metrics: {OUT_DIR / 'q1_metrics_all_od.csv'}")
    print(f"Tops:    {OUT_DIR / 'q1_top_tables.csv'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
