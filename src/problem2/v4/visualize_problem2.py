"""
Problem 2: publication-style figures (English labels, aligned with visualize_problem1).

Inputs:
  - ../data/od_test.csv
  - prediction_week4_v4.csv (ST-GCN)
  - prediction_week4_v4_gru.csv (Pure-GRU, produced by main.py)

Usage:
  python visualize_problem2.py

Outputs: figures_q2/*.png (300 dpi)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
(HERE / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import option as cfg

FIG = cfg.FIG_Q2_DIR
DATA_DIR = HERE.parent / "data"
TEST_CSV = DATA_DIR / "od_test.csv"
PRED_STGCN = Path(cfg.OUT_SUBMISSION)
PRED_GRU = Path(cfg.OUT_SUBMISSION_GRU)

# ---------------------------------------------------------------------------
# figures_q2 统一配色：青绿 Teal + 深紫 Purple + 豆沙粉 Rose（+ 浅粉 Lavender）
# 与 p2_forecast_curves_4od 三线一致；热力图用同系渐变
# ---------------------------------------------------------------------------
P2_TEAL = "#35B0AB"
P2_PURPLE = "#7F4A88"
P2_ROSE = "#DE95BA"
P2_LAVENDER = "#FFD9E8"
P2_BG = "#FFFDFB"
P2_PANEL = "#FFFFFF"
P2_GRID = "#EEE8EE"
P2_SPINE = "#9B8AA0"
P2_EDGE = "#C4B5C0"
P2_TEXT = "#3C4043"
P2_TEXT_MUTED = "#5F6368"
P2_TITLE = "#202124"
P2_LEGEND_EDGE = "#D4C4D0"

# 经典学术对比（折线/柱状，色盲友好，与紫粉系预测曲线图区分）
ACAD_BLUE = "#0072B2"
ACAD_ORANGE = "#E69F00"
ACAD_GRID = "#E5E5E5"
ACAD_SPINE = "#333333"
ACAD_BG = "#FFFFFF"

# 热力图：参考「图三」绿 → 青绿 → 天蓝 → 深蓝
def _cmap_mae_green_blue() -> mpl.colors.LinearSegmentedColormap:
    colors = [
        "#F7FEF0",
        "#E9F7E5",
        "#CEEFCC",
        "#BFE8C1",
        "#BCF4C5",
        "#92C2A6",
        "#D6F6FF",
        "#ACEEFE",
        "#6FC8CA",
        "#58B8D1",
        "#3492B2",
        "#04579B",
    ]
    return mpl.colors.LinearSegmentedColormap.from_list("p2_mae_gnbu", colors, N=256)


# 网络图：亮薄荷 / 天蓝 / 青蓝（同系高亮度）
NET_BRIGHT_LO = "#BCF4C5"
NET_BRIGHT_MID = "#ACEEFE"
NET_BRIGHT_HI = "#58B8D1"
NET_EDGE_LINE = "#3492B2"
NET_NODE_EDGECOLOR = "#04579B"


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
        }
    )


def load_merged() -> pd.DataFrame:
    if not TEST_CSV.exists():
        print(f"Missing {TEST_CSV}", file=sys.stderr)
        sys.exit(1)
    if not PRED_STGCN.exists():
        print(f"Missing {PRED_STGCN}; run main.py first.", file=sys.stderr)
        sys.exit(1)

    truth = pd.read_csv(TEST_CSV, parse_dates=["time_slot"])
    ps = pd.read_csv(PRED_STGCN, parse_dates=["timestamp"])
    ps = ps.rename(columns={"timestamp": "time_slot", "flow_pred": "pred_stgcn"})
    m = truth.merge(
        ps,
        on=["time_slot", "in_station", "out_station"],
        how="inner",
    )
    if PRED_GRU.exists():
        pg = pd.read_csv(PRED_GRU, parse_dates=["timestamp"])
        pg = pg.rename(columns={"timestamp": "time_slot", "flow_pred": "pred_gru"})
        m = m.merge(
            pg,
            on=["time_slot", "in_station", "out_station"],
            how="inner",
        )
    else:
        m["pred_gru"] = np.nan
        print("Warning: GRU predictions not found; curves/ablation use ST-GCN only.", file=sys.stderr)
    m["hour"] = m["time_slot"].dt.hour + m["time_slot"].dt.minute / 60.0
    m["hour_bin"] = m["time_slot"].dt.hour.astype(int)
    return m


def pick_od_panels(df: pd.DataFrame, k: int = 4) -> list[tuple[str, str]]:
    g = df.groupby(["in_station", "out_station"], as_index=False)["flow"].mean()
    g = g.sort_values("flow", ascending=False)
    pairs = [(r.in_station, r.out_station) for r in g.head(k).itertuples()]
    return pairs


def fig_curves_panels(df: pd.DataFrame, pairs: list[tuple[str, str]], path: Path) -> None:
    """2×2 预测曲线：鲜艳学术配色、线型分层。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.35))
    fig.patch.set_facecolor(P2_BG)
    axes = axes.ravel()

    for ax, (o, d) in zip(axes, pairs):
        sub = df[(df["in_station"] == o) & (df["out_station"] == d)].sort_values("time_slot")
        x = np.arange(len(sub))
        ax.set_facecolor(P2_PANEL)
        for spine in ax.spines.values():
            spine.set_color(P2_SPINE)
            spine.set_linewidth(0.8)
        ax.grid(True, color=P2_GRID, linestyle="-", linewidth=0.7, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)

        y_true = sub["flow"].values
        y_s = sub["pred_stgcn"].values
        has_gru = sub["pred_gru"].notna().all()

        # 先画真值（略细、作参照），再消融（虚线），主模型压顶（最醒目）
        ax.plot(
            x,
            y_true,
            color=P2_TEAL,
            label="Ground truth",
            lw=1.5,
            alpha=1.0,
            zorder=2,
            solid_capstyle="round",
        )
        if has_gru:
            ax.plot(
                x,
                sub["pred_gru"].values,
                color=P2_ROSE,
                label="Pure-GRU",
                lw=1.85,
                ls=(0, (5, 2.5)),
                alpha=1.0,
                zorder=3,
                solid_capstyle="round",
            )
        ax.plot(
            x,
            y_s,
            color=P2_PURPLE,
            label="ST-GCN",
            lw=2.15,
            alpha=1.0,
            zorder=4,
            solid_capstyle="round",
        )

        ax.set_title(
            f"{o} $\\to$ {d} (week 4)",
            fontsize=10.5,
            color=P2_TITLE,
            pad=6,
            fontweight="600",
        )
        ax.set_xlabel("Time slot index", fontsize=10, color=P2_TEXT)
        ax.set_ylabel("OD flow", fontsize=10, color=P2_TEXT)
        ax.tick_params(colors=P2_TEXT_MUTED, width=0.6, length=3.5)
        leg = ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=0.98,
            edgecolor=P2_LEGEND_EDGE,
            facecolor="#FFFFFF",
            fontsize=8.75,
            borderpad=0.55,
            handlelength=2.4,
            handletextpad=0.55,
            borderaxespad=0.35,
        )
        for leg_line in leg.get_lines():
            leg_line.set_linewidth(2.0)

    fig.suptitle(
        "Forecast vs. ground truth on representative high-volume OD pairs",
        fontsize=12.5,
        fontweight="600",
        color=P2_TITLE,
        y=1.01,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(path, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close()


def fig_scatter_y_yhat(df: pd.DataFrame, path: Path) -> None:
    y = df["flow"].values.astype(np.float64)
    rng = np.random.default_rng(42)
    n = len(y)
    idx = rng.choice(n, size=min(8000, n), replace=False)
    y_s, p_s = y[idx], df["pred_stgcn"].values[idx].astype(np.float64)
    lim = max(float(np.percentile(y_s, 99.5)), float(np.percentile(p_s, 99.5)), 1.0)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    fig.patch.set_facecolor(P2_BG)
    ax.set_facecolor(P2_PANEL)
    for spine in ax.spines.values():
        spine.set_color(P2_SPINE)
    ax.grid(True, color=P2_GRID, linestyle="-", linewidth=0.6, alpha=1.0)
    ax.set_axisbelow(True)
    ax.scatter(y_s, p_s, s=7, alpha=0.28, c=P2_PURPLE, edgecolors="none")
    ax.plot([0, lim], [0, lim], "--", color=P2_TEAL, lw=1.35, label="$y=x$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed flow $y$", color=P2_TEXT)
    ax.set_ylabel("Predicted flow $\\hat{y}$ (ST-GCN)", color=P2_TEXT)
    ax.set_title("ST-GCN: predicted vs. observed (subsample)", color=P2_TITLE, fontweight="600")
    ax.tick_params(colors=P2_TEXT_MUTED)
    ax.legend(loc="upper left", framealpha=0.96, edgecolor=P2_LEGEND_EDGE)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close()


def fig_residual_by_hour(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["ae_stgcn"] = (df["flow"] - df["pred_stgcn"]).abs()
    if df["pred_gru"].notna().all():
        df["ae_gru"] = (df["flow"] - df["pred_gru"]).abs()
    g = df.groupby("hour_bin", as_index=False).agg(
        mae_stgcn=("ae_stgcn", "mean"),
        **({"mae_gru": ("ae_gru", "mean")} if df["pred_gru"].notna().all() else {}),
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.patch.set_facecolor(ACAD_BG)
    ax.set_facecolor(ACAD_BG)
    for spine in ax.spines.values():
        spine.set_color(ACAD_SPINE)
        spine.set_linewidth(0.9)
    ax.grid(True, color=ACAD_GRID, linestyle="-", linewidth=0.7, alpha=1.0)
    ax.set_axisbelow(True)
    ax.plot(
        g["hour_bin"],
        g["mae_stgcn"],
        "o-",
        color=ACAD_BLUE,
        lw=2.25,
        ms=5.5,
        mfc=ACAD_BLUE,
        mec="white",
        mew=0.55,
        label="ST-GCN MAE",
    )
    if "mae_gru" in g.columns:
        ax.plot(
            g["hour_bin"],
            g["mae_gru"],
            "s-",
            color=ACAD_ORANGE,
            lw=2.15,
            ms=4.8,
            mfc=ACAD_ORANGE,
            mec="white",
            mew=0.55,
            label="Pure-GRU MAE",
        )
    ax.set_xlabel("Hour of day", color=ACAD_SPINE)
    ax.set_ylabel("Mean absolute error", color=ACAD_SPINE)
    ax.set_title(
        "Hourly error profile on test week (all OD pairs)",
        color=ACAD_SPINE,
        fontweight="600",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(colors="#555555")
    ax.legend(loc="upper right", framealpha=0.98, edgecolor="#BBBBBB", fancybox=False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300, facecolor=ACAD_BG)
    plt.close()


def fig_ablation_bars_clean(df: pd.DataFrame, path: Path) -> None:
    y = df["flow"].values.astype(np.float64)
    ps = df["pred_stgcn"].values.astype(np.float64)
    rmse_s = float(np.sqrt(mean_squared_error(y, ps)))
    mae_s = float(mean_absolute_error(y, ps))
    r2_s = float(r2_score(y, ps))

    if df["pred_gru"].notna().all():
        pg = df["pred_gru"].values.astype(np.float64)
        rmse_g = float(np.sqrt(mean_squared_error(y, pg)))
        mae_g = float(mean_absolute_error(y, pg))
        r2_g = float(r2_score(y, pg))
    else:
        rmse_g = mae_g = r2_g = None

    if rmse_g is not None:
        ymax0 = max(rmse_s, mae_s, rmse_g, mae_g) * 1.18
    else:
        ymax0 = max(rmse_s, mae_s) * 1.18

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 4.6))
    fig.patch.set_facecolor(ACAD_BG)
    x0 = np.arange(2)
    w = 0.35
    for a in (ax0, ax1):
        a.set_facecolor(ACAD_BG)
        for spine in a.spines.values():
            spine.set_color(ACAD_SPINE)
            spine.set_linewidth(0.9)
        a.grid(True, axis="y", color=ACAD_GRID, linestyle="-", linewidth=0.7, alpha=1.0)
        a.set_axisbelow(True)
        a.tick_params(colors="#555555")

    ax0.bar(
        x0 - w / 2,
        [rmse_s, mae_s],
        width=w,
        label="ST-GCN",
        color=ACAD_BLUE,
        edgecolor=ACAD_SPINE,
        linewidth=0.45,
    )
    ax0.text(0 - w / 2, rmse_s + 0.03 * ymax0, f"{rmse_s:.3f}", ha="center", va="bottom", fontsize=8)
    ax0.text(1 - w / 2, mae_s + 0.03 * ymax0, f"{mae_s:.3f}", ha="center", va="bottom", fontsize=8)
    if rmse_g is not None:
        ax0.bar(
            x0 + w / 2,
            [rmse_g, mae_g],
            width=w,
            label="Pure-GRU",
            color=ACAD_ORANGE,
            edgecolor=ACAD_SPINE,
            linewidth=0.45,
        )
        ax0.text(0 + w / 2, rmse_g + 0.03 * ymax0, f"{rmse_g:.3f}", ha="center", va="bottom", fontsize=8)
        ax0.text(1 + w / 2, mae_g + 0.03 * ymax0, f"{mae_g:.3f}", ha="center", va="bottom", fontsize=8)
    ax0.set_ylim(0, ymax0)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(["RMSE", "MAE"])
    ax0.set_ylabel("Passengers (same scale as submission)", color=ACAD_SPINE)
    ax0.set_title("Error-scale metrics", color=ACAD_SPINE, fontweight="600")
    ax0.legend(loc="upper right", framealpha=0.98, edgecolor="#BBBBBB", fancybox=False)

    x1 = np.array([0])
    ax1.bar(
        x1 - w / 2,
        [r2_s],
        width=w,
        label="ST-GCN",
        color=ACAD_BLUE,
        edgecolor=ACAD_SPINE,
        linewidth=0.45,
    )
    ax1.text(0 - w / 2, r2_s + 0.03, f"{r2_s:.3f}", ha="center", va="bottom", fontsize=8)
    if r2_g is not None:
        ax1.bar(
            x1 + w / 2,
            [r2_g],
            width=w,
            label="Pure-GRU",
            color=ACAD_ORANGE,
            edgecolor=ACAD_SPINE,
            linewidth=0.45,
        )
        ax1.text(0 + w / 2, r2_g + 0.03, f"{r2_g:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["R$^2$"])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R$^2$", color=ACAD_SPINE)
    ax1.set_title("Coefficient of determination", color=ACAD_SPINE, fontweight="600")
    ax1.legend(loc="lower right", framealpha=0.98, edgecolor="#BBBBBB", fancybox=False)

    fig.suptitle(
        "Ablation: ST-GCN vs. Pure-GRU on test week",
        fontweight="bold",
        color=ACAD_SPINE,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300, facecolor=ACAD_BG)
    plt.close()


def fig_mae_heatmap(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["ae"] = (df["flow"] - df["pred_stgcn"]).abs()
    pivot = df.pivot_table(
        index="in_station",
        columns="out_station",
        values="ae",
        aggfunc="mean",
    )
    nodes = sorted(pivot.index.union(pivot.columns))
    pivot = pivot.reindex(index=nodes, columns=nodes)
    for i, n in enumerate(nodes):
        if n in pivot.index and n in pivot.columns:
            pivot.loc[n, n] = np.nan

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    fig.patch.set_facecolor("#F7FEF0")
    ax.set_facecolor("#F7FEF0")
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=_cmap_mae_green_blue(),
        linewidths=0.55,
        linecolor="white",
        cbar_kws={"label": "Mean |error|"},
        vmin=0,
    )
    ax.set_title(
        "ST-GCN: mean absolute error by OD pair (test week)",
        color=ACAD_SPINE,
        fontweight="600",
    )
    ax.set_xlabel("Destination (out)", color=ACAD_SPINE)
    ax.set_ylabel("Origin (in)", color=ACAD_SPINE)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300, facecolor="#F7FEF0")
    plt.close()


def fig_topology_subgraph(path: Path, df: pd.DataFrame) -> None:
    adj_path = Path(cfg.ADJ_CSV)
    if not adj_path.exists():
        return
    dfa = pd.read_csv(adj_path, index_col=0)
    dfa = dfa.sort_index(axis=0).sort_index(axis=1)
    nodes = list(dfa.index)
    A = dfa.values.astype(int)
    G = nx.from_numpy_array(A)
    mapping = {i: nodes[i] for i in range(len(nodes))}
    G = nx.relabel_nodes(G, mapping)

    out_strength = df.groupby("in_station")["flow"].sum().reindex(nodes).fillna(0).to_numpy()
    sizes = 400 + 1800 * (out_strength / (out_strength.max() + 1e-6))

    pos = nx.spring_layout(G, seed=42, k=0.85, iterations=80)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    fig.patch.set_facecolor("#F7FEF0")
    cmap_nodes = mpl.colors.LinearSegmentedColormap.from_list(
        "p2_nodes_bright",
        [NET_BRIGHT_LO, NET_BRIGHT_MID, NET_BRIGHT_HI],
        N=256,
    )
    smin, smax = float(out_strength.min()), float(out_strength.max())
    if smax <= smin:
        node_rgba = [mpl.colors.to_rgba(NET_BRIGHT_MID)] * len(nodes)
    else:
        norm = mpl.colors.Normalize(vmin=smin, vmax=smax)
        node_rgba = [cmap_nodes(norm(s)) for s in out_strength]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=2.1,
        alpha=0.72,
        edge_color=NET_EDGE_LINE,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=node_rgba,
        alpha=0.95,
        edgecolors=NET_NODE_EDGECOLOR,
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_color=NET_NODE_EDGECOLOR,
    )
    ax.set_title(
        "Toy subway subgraph (node size $\\propto$ week-4 total outbound flow)",
        color=ACAD_SPINE,
        fontweight="600",
        pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300, facecolor="#F7FEF0")
    plt.close()


def main() -> None:
    _setup_style()
    FIG.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    pairs = pick_od_panels(df, k=4)

    fig_curves_panels(df, pairs, FIG / "p2_forecast_curves_4od.png")
    fig_scatter_y_yhat(df, FIG / "p2_scatter_y_vs_yhat_stgcn.png")
    fig_residual_by_hour(df, FIG / "p2_mae_by_hour.png")
    fig_ablation_bars_clean(df, FIG / "p2_ablation_metrics_bar.png")
    fig_mae_heatmap(df, FIG / "p2_mae_heatmap_od.png")
    fig_topology_subgraph(FIG / "p2_network_topology_flowsize.png", df)

    print("Saved figures to", FIG.resolve())


if __name__ == "__main__":
    main()
