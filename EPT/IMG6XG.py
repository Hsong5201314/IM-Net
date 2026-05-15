import pandas as pd
import matplotlib.pyplot as plt

# ================= 全局样式（顶刊标准） =================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 加载数据（请根据实际路径调整）
df_meta = pd.read_csv('simulation_logs/sim_metrics_yelp_meta.csv')
df_meta_epoch = df_meta.groupby('epoch').first().reset_index()

# 创建图形，与图5/图7尺寸一致
fig, ax1 = plt.subplots(figsize=(5, 3.5))
ax2 = ax1.twinx()

# 左轴：干涉能量（线性，突出尖峰）
ax1.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
         color='#2C7FB8', linestyle='-', linewidth=2, label='Interference Energy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Interference Energy (linear)', color='#2C7FB8')
ax1.tick_params(axis='y', labelcolor='#2C7FB8')

# 右轴：梯度余弦相似度
ax2.plot(df_meta_epoch['epoch'], df_meta_epoch['grad_cos_sim'],
         color='#DD8452', linestyle='--', linewidth=2, label='Gradient Cosine Similarity')
ax2.set_ylabel('Cosine Similarity', color='#DD8452')
ax2.tick_params(axis='y', labelcolor='#DD8452')
ax2.set_ylim(-1.0, 0.2)

# 标注峰值点（epoch 11）
peak_idx = df_meta_epoch['interference_energy'].idxmax()
peak_epoch = df_meta_epoch.loc[peak_idx, 'epoch']
peak_interf = df_meta_epoch.loc[peak_idx, 'interference_energy']
peak_cos = df_meta_epoch.loc[peak_idx, 'grad_cos_sim']
ax1.scatter(peak_epoch, peak_interf, color='#2C7FB8', s=50, zorder=5)
ax2.scatter(peak_epoch, peak_cos, color='#DD8452', s=50, zorder=5)

# 使用相对偏移量标注，确保文本在图框内
ax1.annotate(f'Epoch {peak_epoch}',
             xy=(peak_epoch, peak_interf),
             xytext=(15, -15),          # 向右15点，向下15点
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=8, ha='left', va='top')

ax1.grid(True, alpha=0.3, linestyle=':')
# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('figures/Fig6_Conflict_Alignment.pdf')
plt.savefig('figures/Fig6_Conflict_Alignment.png')
plt.close()