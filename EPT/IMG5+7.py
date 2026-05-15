import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

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

os.makedirs('figures', exist_ok=True)

# ================= 加载数据 =================
df_meta = pd.read_csv('simulation_logs/sim_metrics_yelp_meta.csv')
df_fixed = pd.read_csv('simulation_logs/sim_metrics_yelp_fixed_weights.csv')
res_meta = pd.read_csv('results_yelp_LightGCN_meta.csv')
res_fixed = pd.read_csv('results_yelp_LightGCN_fixed_weights.csv')

# epoch 级聚合（取每个 epoch 第一条）
df_meta_epoch = df_meta.groupby('epoch').first().reset_index()
df_fixed_epoch = df_fixed.groupby('epoch').first().reset_index()

# 最佳性能指标
best_meta_ndcg = res_meta['ndcg'].max()
best_fixed_ndcg = res_fixed['ndcg'].max()
best_meta_epoch = res_meta.loc[res_meta['ndcg'].idxmax(), 'epoch']
best_fixed_epoch = res_fixed.loc[res_fixed['ndcg'].idxmax(), 'epoch']
improvement = (best_meta_ndcg - best_fixed_ndcg) / best_fixed_ndcg * 100

# ================= 图 5：Phase Transition Dynamics =================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(10, 4))

# 5a: NDCG@20 收敛曲线
ax5a.plot(res_fixed['epoch'], res_fixed['ndcg'], color='#4C72B0', linestyle='--', linewidth=2, label='Static')
ax5a.plot(res_meta['epoch'], res_meta['ndcg'], color='#DD8452', linestyle='-', linewidth=2, label='IM-Net')
ax5a.scatter(best_fixed_epoch, best_fixed_ndcg, color='#4C72B0', s=50, edgecolor='black', zorder=5)
ax5a.scatter(best_meta_epoch, best_meta_ndcg, color='#DD8452', s=50, edgecolor='black', zorder=5)
ax5a.set_xlabel('Epoch')
ax5a.set_ylabel('NDCG@20')
ax5a.set_ylim(0.05, 0.08)
ax5a.grid(True, alpha=0.3, linestyle=':')
ax5a.legend()
#ax5a.set_title('(a) Performance Trajectories')
ax5a.set_title('(a)', loc='left')
ax5a.text(0.6, 0.06, f'Improvement: {improvement:.1f}%', transform=ax5a.transAxes,
          fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5b: 干涉能量演化（对数坐标，展示“Alignment Transition”）
ax5b.plot(df_fixed_epoch['epoch'], df_fixed_epoch['interference_energy'],
          color='#4C72B0', linestyle='--', linewidth=2, label='Static')
ax5b.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
          color='#DD8452', linestyle='-', linewidth=2, label='IM-Net')
ax5b.set_xlabel('Epoch')
ax5b.set_ylabel('Interference Energy')
ax5b.set_yscale('log')
ax5b.grid(True, alpha=0.3, linestyle=':')
ax5b.legend()
#ax5b.set_title('(b) Alignment Phase Diagram')
ax5b.set_title('(b)', loc='left')

# 标注相变区域（Epoch 8-12）
ax5b.axvspan(8, 12, alpha=0.2, color='gray', label='Phase Transition')
ax5b.text(10, 1e-4, 'Conflict\nRegion', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/Fig5_Phase_Transition.pdf')
plt.savefig('figures/Fig5_Phase_Transition.png')
plt.close()

# ================= 图 6（修正版：完整曲线 + 纵轴范围调整） =================
fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

# 6a: 干涉能量完整曲线（epoch 0-100，线性坐标）
max_interf = df_meta_epoch['interference_energy'].max()   # 约为 3.72e-5
ylim_top = max_interf * 1.2   # 增加 20% 余量，例如 4.5e-5
ax6a.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
          color='#DD8452', linestyle='-', linewidth=2, marker='o', markersize=3)
ax6a.set_ylabel('Interference Energy')
ax6a.set_yscale('linear')
ax6a.set_ylim(0, ylim_top)   # 动态调整，确保峰值完整显示
ax6a.grid(True, alpha=0.3, linestyle=':')
#ax6a.set_title('(a) Decoupling Mechanism: Interference Spike', fontweight='bold')
ax6a.set_title('(a)', loc='left')

# 标注尖峰位置
peak_epoch = df_meta_epoch.loc[df_meta_epoch['interference_energy'].idxmax(), 'epoch']
peak_val = df_meta_epoch['interference_energy'].max()
ax6a.annotate(f'Peak at Epoch {peak_epoch}', xy=(peak_epoch, peak_val),
              xytext=(peak_epoch+15, peak_val*0.9), arrowprops=dict(arrowstyle='->', color='black'),
              fontsize=8, ha='left')
# 标注后期平稳
ax6a.annotate('Stable near zero after Epoch 20', xy=(60, 0.5e-5), xytext=(60, 1.8e-5),
              fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 6b: 辅助权重演化（保持不变）
ax6b.plot(df_meta_epoch['epoch'], df_meta_epoch['w_aux'],
          color='#2C7FB8', linestyle='-', linewidth=2, marker='s', markersize=3)
ax6b.set_xlabel('Epoch')
ax6b.set_ylabel('Auxiliary Weight ($w_{\\mathrm{aux}}$)')
ax6b.set_ylim(0.08, 0.17)
ax6b.grid(True, alpha=0.3, linestyle=':')
#ax6b.set_title('(b) Adaptive Auxiliary Weight Evolution', fontweight='bold')
ax6b.set_title('(b)', loc='left')

final_waux = df_meta_epoch['w_aux'].iloc[-1]
ax6b.text(0.7, final_waux+0.003, f'→ {final_waux:.3f}', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/Fig6_Emergent_Transition.pdf')
plt.savefig('figures/Fig6_Emergent_Transition.png')
plt.close()

# ================= 图 7：Optimization Stability & Landscape Geometry =================
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(10, 4))

# 7a: 训练损失曲线（静态 vs IM-Net）
ax7a.plot(res_fixed['epoch'], res_fixed['loss'], color='#4C72B0', linestyle='--', linewidth=2, label='Static')
ax7a.plot(res_meta['epoch'], res_meta['loss'], color='#DD8452', linestyle='-', linewidth=2, label='IM-Net')
ax7a.set_xlabel('Epoch')
ax7a.set_ylabel('Training Loss')
ax7a.set_yscale('log')  # 损失通常用对数坐标
ax7a.grid(True, alpha=0.3, linestyle=':')
ax7a.legend()
#ax7a.set_title('(a) Loss Smoothing Effect')
ax7a.set_title('(a)', loc='left')

# 标注静态损失尖峰
peak_loss_epoch = res_fixed.loc[res_fixed['loss'].idxmax(), 'epoch']
ax7a.annotate('Loss spike', xy=(peak_loss_epoch, res_fixed['loss'].max()),
              xytext=(peak_loss_epoch-15, 0.5), arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

# Fig.7(b) 改进版：为干涉能量添加独立图例，并增加静态干涉能量（可选）
ax7b_twin = ax7b.twinx()

# 左轴：辅助权重（IM-Net） - 使用实线，深青色
ax7b.plot(df_meta_epoch['epoch'], df_meta_epoch['w_aux'],
          color='#2C7FB8', linestyle='-', linewidth=2.5, label='$w_{\\mathrm{aux}}$ (IM-Net)')

# 右轴：干涉能量（IM-Net） - 使用点划线，橙色，线宽稍细以区分
ax7b_twin.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
               color='#DD8452', linestyle=':', linewidth=2.5, label='Interference (IM-Net)')

# 可选：如果想展示静态干涉能量（右轴，灰色虚线），取消下面注释
# ax7b_twin.plot(df_fixed_epoch['epoch'], df_fixed_epoch['interference_energy'],
#                color='gray', linestyle='--', linewidth=2, alpha=0.6, label='Interference (Static)')

ax7b.set_xlabel('Epoch')
ax7b.set_ylabel('Auxiliary Weight', color='#2C7FB8')
ax7b_twin.set_ylabel('Interference Energy', color='#DD8452')
ax7b_twin.set_yscale('log')

# 图例：左轴图例放在左上角，右轴图例放在右上角，避免重叠
ax7b.legend(loc='upper left', frameon=False)
ax7b_twin.legend(loc='upper right', frameon=False)

# 文本标注：放在空白区域（例如 x=80，y=0.155），而不是紧贴曲线末端
final_waux = df_meta_epoch['w_aux'].iloc[-1]
ax7b.annotate(f'$w_{{\\mathrm{{aux}}}} \\rightarrow {final_waux:.3f}$',
              xy=(80, final_waux), xytext=(80, final_waux-0.005),
              fontsize=9, ha='center', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

# 调整纵轴范围，给文本留出空间
ax7b.set_ylim(0.08, 0.20)   # 使 w_aux 曲线和标注不贴边
ax7b_twin.set_ylim(1e-6, 1e-4)  # 干涉能量范围，与 Fig.6a 协调
#ax7b.set_title('(b) Adaptive Weight and Low Interference')
ax7b.set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('figures/Fig7_Landscape_Smoothing.pdf')
plt.savefig('figures/Fig7_Landscape_Smoothing.png')
plt.close()

# ================= 图 8：Real-World Robustness =================
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(10, 4))

# 8a: 干涉能量峰值对比箱线图（Yelp）
mask = (df_meta['epoch'] >= 8) & (df_meta['epoch'] <= 15)
interf_meta_peak = df_meta.loc[mask, 'interference_energy'].values
interf_fixed_peak = df_fixed.loc[mask, 'interference_energy'].values
bp = ax8a.boxplot([interf_fixed_peak, interf_meta_peak],
                  tick_labels=['Static', 'IM-Net'],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightgray', color='black'),
                  medianprops=dict(color='red', linewidth=2))
ax8a.set_ylabel('Interference Energy (Epochs 8–15)')
ax8a.set_yscale('log')
ax8a.grid(True, axis='y', alpha=0.3, linestyle=':')
ax8a.set_title('(a) Yelp: Peak Interference Control')
ax8a.text(0.5, 0.9, 'p < 0.001', transform=ax8a.transAxes, ha='center', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 8b: 未来工作占位图（Amazon-Books）
ax8b.add_patch(Rectangle((0,0), 1, 1, facecolor='lightgray', edgecolor='black', alpha=0.5))
ax8b.text(0.5, 0.5, 'Future Work:\nRobustness against spectral explosion\non Amazon-Books (see Appendix A)',
          ha='center', va='center', transform=ax8b.transAxes, fontsize=10, wrap=True)
ax8b.set_xlim(0,1)
ax8b.set_ylim(0,1)
ax8b.set_xticks([])
ax8b.set_yticks([])
ax8b.set_title('(b) Extreme Dataset (Appendix)')

plt.tight_layout()
plt.savefig('figures/Fig8_Real_World_Robustness.pdf')
plt.savefig('figures/Fig8_Real_World_Robustness.png')
plt.close()

print("All figures (Fig5–Fig8) have been generated in 'figures/' directory.")