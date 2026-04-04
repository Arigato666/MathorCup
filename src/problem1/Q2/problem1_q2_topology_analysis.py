"""
问题一第二问：网络拓扑属性与客流量的关联分析
======================================================
核心任务：
  1. 计算各节点的度中心性(Degree Centrality)与介数中心性(Betweenness Centrality)
  2. 分析拓扑中心性与节点客流量的相关性（Pearson / Spearman）
  3. 按OD对最短路径跳数分组，比较相邻OD vs 远端OD 的流量波动特征
  4. 可视化以上所有分析结果
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 全局绘图设置（支持中文字体）
# ─────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'   # 如需中文改为 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ═══════════════════════════════════════════════════════
# 0. 数据读取
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("STEP 0: Loading Data")
print("=" * 60)

# --- 邻接矩阵 ---
adj_df = pd.read_csv('./toy_network_adjacency.csv', index_col=0)
nodes = list(adj_df.columns)
N = len(nodes)
A = adj_df.values.astype(int)
print(f"Network nodes ({N}): {nodes}")
print(f"Adjacency matrix shape: {A.shape}")

# --- OD 客流时序数据 ---
flow_df = pd.read_csv('./od_cleaned_full.csv')
# 兼容字段名（赛题说明与示例数据字段略有差异）
flow_df.columns = flow_df.columns.str.strip()
# 时间戳统一
if 'timestamp' in flow_df.columns:
    flow_df['timestamp'] = pd.to_datetime(flow_df['timestamp'])
elif 'time_slot' in flow_df.columns:
    flow_df.rename(columns={'time_slot': 'timestamp'}, inplace=True)
    flow_df['timestamp'] = pd.to_datetime(flow_df['timestamp'])

# 站名列兼容
if 'in_station' not in flow_df.columns and 'origin' in flow_df.columns:
    flow_df.rename(columns={'origin': 'in_station', 'destination': 'out_station'}, inplace=True)

# 去除自身 OD（同站进出）
flow_df = flow_df[flow_df['in_station'] != flow_df['out_station']].copy()

# 缺失值填 0
flow_df['flow'] = flow_df['flow'].fillna(0)

print(f"Flow records: {len(flow_df):,}")
print(f"Date range: {flow_df['timestamp'].min()} ~ {flow_df['timestamp'].max()}")

# ═══════════════════════════════════════════════════════
# 1. 构建 NetworkX 图 & 计算拓扑指标
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Network Topology Metrics")
print("=" * 60)

G = nx.from_numpy_array(A, create_using=nx.Graph())
mapping = {i: nodes[i] for i in range(N)}
G = nx.relabel_nodes(G, mapping)

# ── 1.1 度中心性
# 定义: DC(v) = deg(v) / (N-1)
# 含义: 节点直接相连的邻居比例，越高说明物理连通性越强
degree_centrality = nx.degree_centrality(G)

# ── 1.2 介数中心性
# 定义: BC(v) = Σ_{s≠v≠t} [σ_st(v) / σ_st] / [(N-1)(N-2)/2]
#   σ_st   : s→t 所有最短路径数
#   σ_st(v): 经过节点v的最短路径数
# 含义: 节点作为其他节点间最短路径"中转站"的频率，越高说明是关键枢纽
betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

# ── 1.3 接近中心性
# 定义: CC(v) = (N-1) / Σ_{u≠v} d(v,u)
# 含义: 节点到其他所有节点平均距离的倒数，越高说明地理位置越居中
closeness_centrality = nx.closeness_centrality(G)

# ── 1.4 节点度（原始值）
degree_raw = dict(G.degree())

topo_df = pd.DataFrame({
    'node': nodes,
    'degree': [degree_raw[n] for n in nodes],
    'degree_centrality': [degree_centrality[n] for n in nodes],
    'betweenness_centrality': [betweenness_centrality[n] for n in nodes],
    'closeness_centrality': [closeness_centrality[n] for n in nodes],
})
print(topo_df.to_string(index=False))

# ── 1.5 各 OD 对最短路径跳数
od_path_length = {}
for o in nodes:
    for d in nodes:
        if o != d:
            try:
                od_path_length[(o, d)] = nx.shortest_path_length(G, o, d)
            except nx.NetworkXNoPath:
                od_path_length[(o, d)] = np.inf

# ═══════════════════════════════════════════════════════
# 2. 计算节点级与 OD 级客流统计
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Flow Aggregation")
print("=" * 60)

# ── 2.1 节点总出发流量、到达流量、双向合计
node_out_flow = flow_df.groupby('in_station')['flow'].sum().rename('out_flow')
node_in_flow  = flow_df.groupby('out_station')['flow'].sum().rename('in_flow')
topo_df = topo_df.set_index('node')
topo_df = topo_df.join(node_out_flow, how='left').join(node_in_flow, how='left')
topo_df['total_flow'] = topo_df['out_flow'].fillna(0) + topo_df['in_flow'].fillna(0)
topo_df = topo_df.reset_index().rename(columns={'index': 'node'})
print(topo_df[['node', 'degree', 'degree_centrality',
               'betweenness_centrality', 'total_flow']].to_string(index=False))

# ── 2.2 OD 对统计：均值、标准差、CV
od_stats = flow_df.groupby(['in_station', 'out_station'])['flow'].agg(
    mean_flow='mean',
    std_flow='std',
    total_flow='sum',
    nonzero_ratio=lambda x: (x > 0).mean()   # 非零占比（稀疏程度）
).reset_index()
od_stats['cv'] = od_stats['std_flow'] / (od_stats['mean_flow'] + 1e-9)  # 变异系数
od_stats['hop'] = od_stats.apply(
    lambda r: od_path_length.get((r['in_station'], r['out_station']), np.inf), axis=1
)
od_stats['hop'] = od_stats['hop'].replace(np.inf, np.nan).dropna()
od_stats = od_stats.dropna(subset=['hop'])
od_stats['hop'] = od_stats['hop'].astype(int)

print(f"\nOD pairs stats:\n{od_stats.groupby('hop')[['mean_flow','std_flow','cv','nonzero_ratio']].mean().round(3)}")

# ═══════════════════════════════════════════════════════
# 3. 相关性分析：拓扑中心性 ↔ 客流量
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Correlation Analysis")
print("=" * 60)

metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
corr_results = []
for m in metrics:
    r_p, p_p = stats.pearsonr(topo_df[m], topo_df['total_flow'])
    r_s, p_s = stats.spearmanr(topo_df[m], topo_df['total_flow'])
    corr_results.append({'metric': m,
                         'Pearson_r': round(r_p, 3), 'Pearson_p': round(p_p, 4),
                         'Spearman_r': round(r_s, 3), 'Spearman_p': round(p_s, 4)})
    print(f"{m:30s}  Pearson r={r_p:.3f}(p={p_p:.3f})  Spearman r={r_s:.3f}(p={p_s:.3f})")

corr_df = pd.DataFrame(corr_results)

# ═══════════════════════════════════════════════════════
# 4. 按跳数分组的波动特征检验
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Hop-distance vs Flow Volatility")
print("=" * 60)

hop_groups = od_stats.groupby('hop')
hop_summary = hop_groups[['mean_flow', 'std_flow', 'cv', 'nonzero_ratio']].mean().round(4)
print(hop_summary)

# Kruskal-Wallis 检验：不同跳数组的 CV 是否显著差异
groups_cv = [g['cv'].values for _, g in hop_groups if len(g) > 1]
if len(groups_cv) >= 2:
    kw_stat, kw_p = stats.kruskal(*groups_cv)
    print(f"\nKruskal-Wallis test on CV across hop groups: H={kw_stat:.3f}, p={kw_p:.4f}")
    sig = "SIGNIFICANT" if kw_p < 0.05 else "not significant"
    print(f"  → Difference is {sig} (alpha=0.05)")

# ═══════════════════════════════════════════════════════
# 5. 可视化（共 4 张图，存为一张大图）
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Visualization")
print("=" * 60)

fig = plt.figure(figsize=(20, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

BLUE   = '#2E86AB'
ORANGE = '#F4A261'
GREEN  = '#2A9D8F'
RED    = '#E76F51'
PURPLE = '#9B5DE5'
GRAY   = '#8D99AE'

# ──────────────────────────────────────────────────────
# 图1: 网络拓扑图（节点大小=度, 节点颜色=介数中心性）
# ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
pos = nx.spring_layout(G, seed=42, k=1.5)
node_sizes  = [degree_raw[n] * 400 + 300 for n in G.nodes()]
node_colors = [betweenness_centrality[n] for n in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=GRAY, width=2.0, alpha=0.7)
sc = nx.draw_networkx_nodes(G, pos, ax=ax1,
                             node_size=node_sizes,
                             node_color=node_colors,
                             cmap=plt.cm.YlOrRd, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9, font_weight='bold')
plt.colorbar(sc, ax=ax1, label='Betweenness Centrality', shrink=0.8)
ax1.set_title('Network Topology\n(size=degree, color=betweenness)', fontsize=11, fontweight='bold')
ax1.axis('off')

# ──────────────────────────────────────────────────────
# 图2: 度中心性 vs 总客流量（散点 + 回归线）
# ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
x2 = topo_df['degree_centrality'].values
y2 = topo_df['total_flow'].values
ax2.scatter(x2, y2, s=120, color=BLUE, zorder=5, edgecolors='white', linewidth=0.8)
for _, row in topo_df.iterrows():
    ax2.annotate(row['node'], (row['degree_centrality'], row['total_flow']),
                 textcoords='offset points', xytext=(5, 3), fontsize=8, color='#333333')
# 线性回归
m2, b2, r2_, _, _ = stats.linregress(x2, y2)
xfit = np.linspace(x2.min(), x2.max(), 100)
ax2.plot(xfit, m2 * xfit + b2, '--', color=RED, linewidth=1.5, label=f'r={corr_df.loc[0,"Pearson_r"]:.2f}')
ax2.set_xlabel('Degree Centrality', fontsize=10)
ax2.set_ylabel('Total Flow (all time)', fontsize=10)
ax2.set_title('Degree Centrality vs Total Flow', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────
# 图3: 介数中心性 vs 总客流量
# ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
x3 = topo_df['betweenness_centrality'].values
y3 = topo_df['total_flow'].values
ax3.scatter(x3, y3, s=120, color=GREEN, zorder=5, edgecolors='white', linewidth=0.8)
for _, row in topo_df.iterrows():
    ax3.annotate(row['node'], (row['betweenness_centrality'], row['total_flow']),
                 textcoords='offset points', xytext=(5, 3), fontsize=8, color='#333333')
m3, b3, _, _, _ = stats.linregress(x3, y3)
xfit3 = np.linspace(x3.min(), x3.max(), 100)
ax3.plot(xfit3, m3 * xfit3 + b3, '--', color=RED, linewidth=1.5,
         label=f'r={corr_df.loc[1,"Pearson_r"]:.2f}')
ax3.set_xlabel('Betweenness Centrality', fontsize=10)
ax3.set_ylabel('Total Flow (all time)', fontsize=10)
ax3.set_title('Betweenness Centrality vs Total Flow', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────
# 图4: 三种中心性的相关系数对比（柱状图）
# ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
metric_labels = ['Degree\nCentrality', 'Betweenness\nCentrality', 'Closeness\nCentrality']
pearson_vals  = corr_df['Pearson_r'].values
spearman_vals = corr_df['Spearman_r'].values
x4 = np.arange(len(metric_labels))
w = 0.35
bars1 = ax4.bar(x4 - w/2, pearson_vals,  width=w, color=BLUE,   label='Pearson r',  alpha=0.85)
bars2 = ax4.bar(x4 + w/2, spearman_vals, width=w, color=ORANGE, label='Spearman r', alpha=0.85)
ax4.set_xticks(x4)
ax4.set_xticklabels(metric_labels, fontsize=9)
ax4.set_ylabel('Correlation Coefficient', fontsize=10)
ax4.set_ylim(-1, 1)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_title('Centrality–Flow Correlation\n(Pearson & Spearman)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, axis='y', alpha=0.3)
# 标注 p 值显著性
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    pv = corr_df.loc[i, 'Pearson_p']
    marker = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns'))
    ax4.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
             marker, ha='center', va='bottom', fontsize=8, color=BLUE)

# ──────────────────────────────────────────────────────
# 图5: 按跳数分组的箱线图（均值流量）
# ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
hop_vals = sorted(od_stats['hop'].unique())
data_by_hop = [od_stats[od_stats['hop'] == h]['mean_flow'].values for h in hop_vals]
bp = ax5.boxplot(data_by_hop, patch_artist=True, notch=False,
                 medianprops={'color': 'black', 'linewidth': 1.5})
colors_box = [BLUE, GREEN, ORANGE, RED, PURPLE][:len(hop_vals)]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_xticks(range(1, len(hop_vals) + 1))
ax5.set_xticklabels([f'hop={h}' for h in hop_vals], fontsize=9)
ax5.set_ylabel('Mean Flow per Time Slot', fontsize=10)
ax5.set_title('Mean Flow Distribution by Hop Distance', fontsize=11, fontweight='bold')
ax5.grid(True, axis='y', alpha=0.3)

# ──────────────────────────────────────────────────────
# 图6: 按跳数分组的变异系数箱线图（CV = std/mean）
# ──────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cv_by_hop = [od_stats[od_stats['hop'] == h]['cv'].values for h in hop_vals]
bp2 = ax6.boxplot(cv_by_hop, patch_artist=True,
                  medianprops={'color': 'black', 'linewidth': 1.5})
for patch, color in zip(bp2['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_xticks(range(1, len(hop_vals) + 1))
ax6.set_xticklabels([f'hop={h}' for h in hop_vals], fontsize=9)
ax6.set_ylabel('CV  (std / mean)', fontsize=10)
ax6.set_title('Flow Volatility (CV) by Hop Distance\n(higher CV = more irregular)', fontsize=11, fontweight='bold')
ax6.grid(True, axis='y', alpha=0.3)

# ──────────────────────────────────────────────────────
# 图7: 跳数 vs 非零比例（稀疏程度）
# ──────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
hop_mean_nz = od_stats.groupby('hop')['nonzero_ratio'].mean()
ax7.bar([f'hop={h}' for h in hop_mean_nz.index], hop_mean_nz.values,
        color=colors_box, alpha=0.8, edgecolor='white')
ax7.set_ylabel('Avg Non-zero Ratio', fontsize=10)
ax7.set_ylim(0, 1)
ax7.set_title('Sparsity by Hop Distance\n(non-zero ratio: higher = denser)', fontsize=11, fontweight='bold')
ax7.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(hop_mean_nz.values):
    ax7.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

# ──────────────────────────────────────────────────────
# 图8: OD 客流热力图（平均流量矩阵）
# ──────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1:])
od_matrix = od_stats.pivot_table(index='in_station', columns='out_station',
                                  values='mean_flow', fill_value=0)
od_matrix = od_matrix.reindex(index=nodes, columns=nodes, fill_value=0)
mask = np.eye(len(nodes), dtype=bool)  # 对角线遮蔽
sns.heatmap(od_matrix, ax=ax8, cmap='YlOrRd', annot=True, fmt='.1f',
            mask=mask, linewidths=0.4, linecolor='white',
            cbar_kws={'label': 'Mean Flow / time slot', 'shrink': 0.8})
ax8.set_title('OD Mean Flow Heatmap\n(row=Origin, col=Destination)', fontsize=11, fontweight='bold')
ax8.set_xlabel('Destination', fontsize=10)
ax8.set_ylabel('Origin', fontsize=10)
ax8.tick_params(axis='x', rotation=30)
ax8.tick_params(axis='y', rotation=0)

# ── 总标题
fig.suptitle('Problem 1-Q2: Network Topology vs Passenger Flow Analysis',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('./topology_flow_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("\n[Saved] topology_flow_analysis.png")

# ──────────────────────────────────────────────────────
# 保存各个独立子图
# ──────────────────────────────────────────────────────
print("Saving individual figures...")

# 图1
fig_1, ax_1 = plt.subplots(figsize=(8, 6))
nx.draw_networkx_edges(G, pos, ax=ax_1, edge_color=GRAY, width=2.0, alpha=0.7)
sc1 = nx.draw_networkx_nodes(G, pos, ax=ax_1, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.YlOrRd, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax_1, font_size=9, font_weight='bold')
fig_1.colorbar(sc1, ax=ax_1, label='Betweenness Centrality', shrink=0.8)
ax_1.set_title('Network Topology\n(size=degree, color=betweenness)', fontsize=11, fontweight='bold')
ax_1.axis('off')
fig_1.savefig('./fig1.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_1)

# 图2
fig_2, ax_2 = plt.subplots(figsize=(8, 6))
ax_2.scatter(x2, y2, s=120, color=BLUE, zorder=5, edgecolors='white', linewidth=0.8)
for _, row in topo_df.iterrows():
    ax_2.annotate(row['node'], (row['degree_centrality'], row['total_flow']), textcoords='offset points', xytext=(5, 3), fontsize=8, color='#333333')
ax_2.plot(xfit, m2 * xfit + b2, '--', color=RED, linewidth=1.5, label=f'r={corr_df.loc[0,"Pearson_r"]:.2f}')
ax_2.set_xlabel('Degree Centrality', fontsize=10)
ax_2.set_ylabel('Total Flow (all time)', fontsize=10)
ax_2.set_title('Degree Centrality vs Total Flow', fontsize=11, fontweight='bold')
ax_2.legend(fontsize=9)
ax_2.grid(True, alpha=0.3)
fig_2.savefig('./fig2.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_2)

# 图3
fig_3, ax_3 = plt.subplots(figsize=(8, 6))
ax_3.scatter(x3, y3, s=120, color=GREEN, zorder=5, edgecolors='white', linewidth=0.8)
for _, row in topo_df.iterrows():
    ax_3.annotate(row['node'], (row['betweenness_centrality'], row['total_flow']), textcoords='offset points', xytext=(5, 3), fontsize=8, color='#333333')
ax_3.plot(xfit3, m3 * xfit3 + b3, '--', color=RED, linewidth=1.5, label=f'r={corr_df.loc[1,"Pearson_r"]:.2f}')
ax_3.set_xlabel('Betweenness Centrality', fontsize=10)
ax_3.set_ylabel('Total Flow (all time)', fontsize=10)
ax_3.set_title('Betweenness Centrality vs Total Flow', fontsize=11, fontweight='bold')
ax_3.legend(fontsize=9)
ax_3.grid(True, alpha=0.3)
fig_3.savefig('./fig3.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_3)

# 图4
fig_4, ax_4 = plt.subplots(figsize=(8, 6))
bars1_4 = ax_4.bar(x4 - w/2, pearson_vals, width=w, color=BLUE, label='Pearson r', alpha=0.85)
bars2_4 = ax_4.bar(x4 + w/2, spearman_vals, width=w, color=ORANGE, label='Spearman r', alpha=0.85)
ax_4.set_xticks(x4)
ax_4.set_xticklabels(metric_labels, fontsize=9)
ax_4.set_ylabel('Correlation Coefficient', fontsize=10)
ax_4.set_ylim(-1, 1)
ax_4.axhline(0, color='black', linewidth=0.8)
ax_4.set_title('Centrality-Flow Correlation (Pearson & Spearman)', fontsize=11, fontweight='bold')
ax_4.legend(fontsize=9)
ax_4.grid(True, axis='y', alpha=0.3)
for i, (bar1_4, bar2_4) in enumerate(zip(bars1_4, bars2_4)):
    pv = corr_df.loc[i, 'Pearson_p']
    marker = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns'))
    ax_4.text(bar1_4.get_x() + bar1_4.get_width()/2, bar1_4.get_height() + 0.02, marker, ha='center', va='bottom', fontsize=8, color=BLUE)
fig_4.savefig('./fig4.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_4)

# 图5
fig_5, ax_5 = plt.subplots(figsize=(8, 6))
bp_5 = ax_5.boxplot(data_by_hop, patch_artist=True, notch=False, medianprops={'color': 'black', 'linewidth': 1.5})
for patch, color in zip(bp_5['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_5.set_xticks(range(1, len(hop_vals) + 1))
ax_5.set_xticklabels([f'hop={h}' for h in hop_vals], fontsize=9)
ax_5.set_ylabel('Mean Flow per Time Slot', fontsize=10)
ax_5.set_title('Mean Flow Distribution by Hop Distance', fontsize=11, fontweight='bold')
ax_5.grid(True, axis='y', alpha=0.3)
fig_5.savefig('./fig5.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_5)

# 图6
fig_6, ax_6 = plt.subplots(figsize=(8, 6))
bp2_6 = ax_6.boxplot(cv_by_hop, patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.5})
for patch, color in zip(bp2_6['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_6.set_xticks(range(1, len(hop_vals) + 1))
ax_6.set_xticklabels([f'hop={h}' for h in hop_vals], fontsize=9)
ax_6.set_ylabel('CV (std / mean)', fontsize=10)
ax_6.set_title('Flow Volatility (CV) by Hop Distance', fontsize=11, fontweight='bold')
ax_6.grid(True, axis='y', alpha=0.3)
fig_6.savefig('./fig6.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_6)

# 图7
fig_7, ax_7 = plt.subplots(figsize=(8, 6))
ax_7.bar([f'hop={h}' for h in hop_mean_nz.index], hop_mean_nz.values, color=colors_box, alpha=0.8, edgecolor='white')
ax_7.set_ylabel('Avg Non-zero Ratio', fontsize=10)
ax_7.set_ylim(0, 1)
ax_7.set_title('Sparsity by Hop Distance', fontsize=11, fontweight='bold')
ax_7.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(hop_mean_nz.values):
    ax_7.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
fig_7.savefig('./fig7.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_7)

# 图8
fig_8, ax_8 = plt.subplots(figsize=(10, 8))
sns.heatmap(od_matrix, ax=ax_8, cmap='YlOrRd', annot=True, fmt='.1f', mask=mask, linewidths=0.4, linecolor='white', cbar_kws={'label': 'Mean Flow / time slot', 'shrink': 0.8})
ax_8.set_title('OD Mean Flow Heatmap', fontsize=11, fontweight='bold')
ax_8.set_xlabel('Destination', fontsize=10)
ax_8.set_ylabel('Origin', fontsize=10)
ax_8.tick_params(axis='x', rotation=30)
ax_8.tick_params(axis='y', rotation=0)
fig_8.savefig('./fig8.jpg', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_8)
print("[Saved] 8 individual figures successfully.\n")

plt.show()

# ═══════════════════════════════════════════════════════
# 6. 输出结论性汇总表
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY TABLE: Topology Metrics per Node")
print("=" * 60)
print(topo_df[['node', 'degree', 'degree_centrality',
               'betweenness_centrality', 'closeness_centrality',
               'total_flow']].to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY TABLE: Flow Statistics by Hop Distance")
print("=" * 60)
print(hop_summary.to_string())

print("\n" + "=" * 60)
print("SUMMARY TABLE: Centrality–Flow Correlation")
print("=" * 60)
print(corr_df.to_string(index=False))

print("\n[Done] All analysis complete.")
