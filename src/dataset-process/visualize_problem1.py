"""
Problem 1 (1): publication-style figures (English labels, clean matplotlib look).

Requires: output/od_cleaned_full.csv and od_metrics_summary.csv from process_od_data.py

Usage:
  python visualize_problem1.py

Outputs: output/figures/*.png (300 dpi)
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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import correlate

OUT = HERE / "output"
FIG = OUT / "figures"

EPS = 1e-5

# Reference style: blue weekday / red weekend, soft peak bands
C_WD = "#2166AC"
C_WE = "#D6604D"
C_AM = "#F5DEB3"
C_PM = "#D8BFD8"


def _setup_style_english() -> None:
    """Clean academic look; no CJK font so labels render consistently everywhere."""
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_path = OUT / "od_cleaned_full.csv"
    met_path = OUT / "od_metrics_summary.csv"
    if not full_path.exists() or not met_path.exists():
        print("Run first: python process_od_data.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(full_path, parse_dates=["time_slot"])
    met = pd.read_csv(met_path)
    return df, met


def fig1_weekday_weekend_curves(df: pd.DataFrame, met: pd.DataFrame) -> None:
    """2x2 OD curves: no smoothing, 0-24 h, per-axis legend + peak bands (reference style)."""
    plot_pairs = [
        ("Node_1", "Node_4"),
        ("Node_2", "Node_5"),
        ("Node_3", "Node_4"),
        ("Node_4", "Node_7"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    fig.suptitle(
        "OD Flow Temporal Pattern Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for ax, (o, d) in zip(axes.flatten(), plot_pairs):
        sub = df[(df["in_station"] == o) & (df["out_station"] == d)].copy()
        sub["t"] = sub["hour"] + sub["minute"] / 60.0

        wd = sub[~sub["is_weekend"]].groupby("t")["flow"].mean()
        we = sub[sub["is_weekend"]].groupby("t")["flow"].mean()
        t_index = np.sort(sub["t"].unique())
        wd = wd.reindex(t_index, fill_value=0).values.astype(float)
        we = we.reindex(t_index, fill_value=0).values.astype(float)

        ax.axvspan(7, 9, alpha=0.35, color=C_AM, zorder=0, label="AM Peak")
        ax.axvspan(17, 19, alpha=0.35, color=C_PM, zorder=0, label="PM Peak")
        ax.plot(t_index, wd, color=C_WD, lw=2.0, label="Weekday", zorder=2)
        ax.plot(t_index, we, color=C_WE, lw=1.8, ls="--", label="Weekend", zorder=2)

        pk = int(np.argmax(wd))
        ax.scatter(
            t_index[pk],
            wd[pk],
            color=C_WD,
            s=28,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

        row_m = met[(met["in_station"] == o) & (met["out_station"] == d)]
        tii = row_m["TII"].values[0] if len(row_m) else float("nan")
        wdi = row_m["WDI"].values[0] if len(row_m) else float("nan")
        ax.set_title(f"{o} -> {d}    TII={tii:.3f}    WDI={wdi:.3f}", fontsize=10)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Flow")
        ax.set_xlim(0.0, 24.0)
        ax.set_xticks(np.arange(0, 25, 2))
        h1, l1 = ax.get_legend_handles_labels()
        ax.legend(h1, l1, loc="upper right", fontsize=7.5, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG / "fig1_weekday_weekend_od_curves.png", bbox_inches="tight")
    plt.close()


def fig2_tii_bars(met: pd.DataFrame) -> None:
    plot_df = met.assign(OD=met["in_station"] + "->" + met["out_station"]).sort_values("TII")
    vals = plot_df["TII"].values
    colors = np.where(vals >= 0, "#2E75B6", "#C00000")

    fig, ax = plt.subplots(figsize=(9.8, 8.5))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.9, height=0.72)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["OD"].values, fontsize=7.5)
    ax.set_xlabel("Tidal Intensity Index (TII)")
    ax.set_title(
        "TII by OD pair (weekday AM vs PM; blue: AM-heavy, red: PM-heavy)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.35)

    top_idx = np.argsort(np.abs(vals))[::-1][:12]
    for i in top_idx:
        v = vals[i]
        ax.text(
            v + (0.02 if v >= 0 else -0.02),
            i,
            f"{v:.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7.5,
            color="#333333",
        )
    plt.tight_layout()
    plt.savefig(FIG / "fig2_tii_all_od_pairs.png", bbox_inches="tight")
    plt.close()


def fig3_heatmap_hour_weekday(df: pd.DataFrame) -> None:
    """Weekday: Mon-Fri rows only; Weekend: Sat-Sun only; shared color scale."""
    wd = df[~df["is_weekend"]].copy()
    we = df[df["is_weekend"]].copy()

    heat_wd = (
        wd.groupby(["weekday", "hour"])["flow"]
        .mean()
        .unstack("hour")
        .reindex(index=[0, 1, 2, 3, 4])
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )
    heat_we = (
        we.groupby(["weekday", "hour"])["flow"]
        .mean()
        .unstack("hour")
        .reindex(index=[5, 6])
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )

    vmax = float(max(heat_wd.values.max(), heat_we.values.max()))
    vmin = 0.0

    day_wd = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    day_we = ["Sat", "Sun"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2))

    for ax, heat, title, ylabels in zip(
        axes,
        [heat_wd, heat_we],
        ["Weekday Heatmap", "Weekend Heatmap"],
        [day_wd, day_we],
    ):
        sns.heatmap(
            heat.values,
            ax=ax,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Avg Flow", "shrink": 0.9},
            linewidths=0,
            xticklabels=2,
            yticklabels=ylabels,
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        ax.set_title(title, fontsize=11.5, fontweight="bold")

    plt.suptitle(
        "Network-wide average flow (all OD pairs summed per slot)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIG / "fig3_heatmap_hour_weekday.png", bbox_inches="tight")
    plt.close()


def fig4_cv_top20(df: pd.DataFrame) -> None:
    cv_wd = (
        df[~df["is_weekend"]]
        .groupby(["in_station", "out_station"])["flow"]
        .agg(lambda x: float(x.std() / (x.mean() + EPS)))
        .reset_index()
        .rename(columns={"flow": "CV_weekday"})
    )
    cv_we = (
        df[df["is_weekend"]]
        .groupby(["in_station", "out_station"])["flow"]
        .agg(lambda x: float(x.std() / (x.mean() + EPS)))
        .reset_index()
        .rename(columns={"flow": "CV_weekend"})
    )
    cv_df = cv_wd.merge(cv_we, on=["in_station", "out_station"])
    cv_df["OD"] = cv_df["in_station"] + "->" + cv_df["out_station"]
    cv_df = cv_df.sort_values("CV_weekday", ascending=False).head(20)

    long_df = cv_df.melt(
        id_vars="OD",
        value_vars=["CV_weekday", "CV_weekend"],
        var_name="type",
        value_name="CV",
    )
    long_df["type"] = long_df["type"].map({"CV_weekday": "Weekday", "CV_weekend": "Weekend"})

    od_order = cv_df["OD"].tolist()
    plt.figure(figsize=(11.5, 5.0))
    ax = sns.barplot(
        data=long_df,
        x="OD",
        y="CV",
        hue="type",
        order=od_order,
        palette=[C_WD, C_WE],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.set_xlabel("OD pair")
    ax.set_title(
        "Weekday vs Weekend CV (Top-20 by weekday CV)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(title=None, frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(np.arange(len(od_order)))
    ax.set_xticklabels(od_order, rotation=45, ha="right", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_cv_weekday_weekend_top20.png", bbox_inches="tight")
    plt.close()


def fig5_weekday_weekend_scatter(met: pd.DataFrame) -> None:
    """Reference style: weekday mean vs weekend mean, WDI color, y=x line."""
    x = met["weekday_mean"].values
    y = met["weekend_mean"].values
    wdi = met["WDI"].values

    lim = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.08
    lim = max(lim, 1.0)
    wabs = max(float(np.nanmax(np.abs(wdi))), 1e-6)

    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    sc = ax.scatter(
        x,
        y,
        c=wdi,
        cmap="RdYlBu_r",
        vmin=-wabs,
        vmax=wabs,
        s=55,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.plot([0, lim], [0, lim], color="gray", ls="--", lw=1.2, alpha=0.75, label="y=x (equal)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Weekday Avg Flow")
    ax.set_ylabel("Weekend Avg Flow")
    ax.set_title(
        "Weekday vs Weekend Flow per OD Pair\n"
        "(above y=x: weekend-dominant; below: weekday-dominant)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.35)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("WDI (Weekday-Weekend Index)")
    plt.tight_layout()
    plt.savefig(FIG / "fig5_weekday_weekend_flow_scatter.png", bbox_inches="tight")
    plt.close()


def fig6_mirror_cc(df: pd.DataFrame, o: str, d: str, max_lag_hours: float = 12.0) -> None:
    wd = df[~df["is_weekend"]].copy()
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
    seq_od = seq_od.reindex(idx, fill_value=0).sort_index()
    seq_do = seq_do.reindex(idx, fill_value=0).sort_index()
    v_od = seq_od.values.astype(float)
    v_do = seq_do.values.astype(float)
    if len(v_od) < 8 or len(v_do) < 8:
        return
    n_od = (v_od - v_od.mean()) / (v_od.std() + EPS)
    n_do = (v_do - v_do.mean()) / (v_do.std() + EPS)
    corr = correlate(n_od, n_do, mode="full")
    lags = np.arange(-(len(n_od) - 1), len(n_od)) * 15.0
    mask = np.abs(lags) <= max_lag_hours * 60.0
    sub_c, sub_l = corr[mask], lags[mask]
    best_i = int(np.argmax(sub_c))
    best_lag = sub_l[best_i]
    best_r = float(sub_c[best_i] / max(len(n_od), 1))

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(sub_l, sub_c / max(len(n_od), 1), color=C_WD, lw=1.9)
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.scatter([best_lag], [best_r], color=C_WE, s=50, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Lag (minutes; negative: D->O leads O->D)")
    ax.set_ylabel("Normalized cross-correlation")
    ax.set_title(
        f"Reverse-OD cross-correlation (weekdays): {o}->{d} vs {d}->{o}\n"
        f"peak r={best_r:.3f} at lag {best_lag:.0f} min",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(alpha=0.35)
    plt.tight_layout()
    safe = f"{o}_{d}".replace(" ", "")
    plt.savefig(FIG / f"fig6_mirror_cc_{safe}.png", bbox_inches="tight")
    plt.close()


def main() -> int:
    _setup_style_english()
    FIG.mkdir(parents=True, exist_ok=True)

    df, met = load_data()
    print(f"Loaded {len(df)} rows, {len(met)} OD metrics")

    fig1_weekday_weekend_curves(df, met)
    print("Saved fig1_weekday_weekend_od_curves.png")
    fig2_tii_bars(met)
    print("Saved fig2_tii_all_od_pairs.png")
    fig3_heatmap_hour_weekday(df)
    print("Saved fig3_heatmap_hour_weekday.png")
    fig4_cv_top20(df)
    print("Saved fig4_cv_weekday_weekend_top20.png")
    fig5_weekday_weekend_scatter(met)
    print("Saved fig5_weekday_weekend_flow_scatter.png")
    fig6_mirror_cc(df, "Node_3", "Node_4")
    print("Saved fig6_mirror_cc_Node_3_Node_4.png")

    print(f"\nAll figures: {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
