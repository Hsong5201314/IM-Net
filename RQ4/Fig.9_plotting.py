import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置期刊风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['pdf.fonttype'] = 42

# 读取 CSV，自动检测分隔符（处理制表符/空格），并去除列名首尾空格
df = pd.read_csv('rq4_stress_test_data.csv', sep=None, engine='python')
df.columns = df.columns.str.strip()

# 如果列名仍未正确分割（可能因为第一行有特殊字符），手动修正
if len(df.columns) == 1 and 'Method' not in df.columns:
    # 读取原始文件第一行，按空白符分割得到列名
    with open('rq4_stress_test_data.csv', 'r') as f:
        header = f.readline().strip()
    import re
    col_names = re.split(r'\s+', header)
    # 重新读取数据，跳过第一行，指定列名
    df = pd.read_csv('rq4_stress_test_data.csv', sep=None, engine='python',
                     names=col_names, skiprows=1)
    df.columns = df.columns.str.strip()

# 去除 Method 列中多余的空格（例如 "IM-Net  (Proactive)" 中的双空格）
df['Method'] = df['Method'].str.strip()

# 打印确认信息
print("列名:", df.columns.tolist())
print("方法名:", df['Method'].unique())
print(df.head(2))

# 动态分配颜色（避免硬编码键名不匹配）
methods = df['Method'].unique()
colors = {m: '#E53935' if 'Proactive' in m else '#B0BEC5' for m in methods}

# 创建图形
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

# (a) 性能轨迹
sns.lineplot(data=df, x='Epoch', y='Recall', hue='Method', palette=colors, lw=2.5, ax=axes[0])
#axes[0].set_title('a. Performance Under Stress (NCF)', fontweight='bold', fontsize=14)
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[0].set_ylabel('Validation Recall@20', fontsize=12)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylim(bottom=0)
axes[0].legend(title='Method', frameon=True)

# (b) 有效梯度对齐轨迹
if 'EffectiveAlignment' in df.columns:
    align_col = 'EffectiveAlignment'
    ylabel = 'Cosine Similarity ρ(g_model, g_main)'
else:
    align_col = 'GradientAlignment'
    ylabel = 'Cosine Similarity ρ(g_main, g_conflict)'

sns.lineplot(data=df, x='Epoch', y=align_col, hue='Method', palette=colors, lw=2.5, ax=axes[1])
#axes[1].set_title('b. Effective Gradient Alignment', fontweight='bold', fontsize=14)
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[1].set_ylabel(ylabel, fontsize=12)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].axhline(0.0, color='grey', linestyle='--', alpha=0.5, lw=1.5)
axes[1].axhline(-1.0, color='lightgrey', linestyle=':', alpha=0.5, lw=1)
axes[1].legend().remove()

# (c) 有害任务权重轨迹（仅 IM-Net）
df_imnet = df[df['Method'] == 'IM-Net (Proactive)']
axes[2].plot(df_imnet['Epoch'], df_imnet['AuxWeight'], color=colors['IM-Net (Proactive)'], lw=2.5)
#axes[2].set_title('c. IM-Net: Harmful Task Weight', fontweight='bold', fontsize=14)
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

axes[2].set_ylabel('Weight of Conflicting Task', fontsize=12)
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylim(0, 1)
axes[2].axhline(0.01, color='red', linestyle='--', alpha=0.7, lw=1.5, label='Lower bound (0.01)')
axes[2].legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('Fig_RQ4_Stress_Test.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Fig_RQ4_Stress_Test.png', dpi=300, bbox_inches='tight')
print("[INFO] Figures saved: Fig_RQ4_Stress_Test.pdf / .png")
plt.show()