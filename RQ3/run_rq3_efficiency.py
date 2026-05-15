import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from data_loaderMeta import DataProcessor, RecDataset, DataLoader
from backbone import LightGCN
from train_engine import IMNetTrainer


# ================= 实验配置生成器（支持少样本、冷启动） =================
def prepare_experiment_data(dp, setting='full', cold_start_ratio=0.2):
    """
    根据实验协议修改 DataProcessor 中的训练数据，并同步裁剪测试集
    setting: 'full', 'few_10', 'few_20', 'few_50', 'cold_user', 'cold_item'
    返回修改后的 dp（直接修改传入的 dp）
    """
    # 获取原始训练数据（list of [user, item]）
    train_data = dp.train_data.copy()

    if setting == 'full':
        pass
    elif setting.startswith('few_'):
        ratio = float(setting.split('_')[1]) / 100.0
        indices = np.random.choice(len(train_data), int(len(train_data) * ratio), replace=False)
        train_data = [train_data[i] for i in indices]
        print(f"[Data] Few-shot {ratio * 100:.0f}%: {len(train_data)} interactions")
    elif setting == 'cold_user':
        users = list(set([u for u, _ in train_data]))
        n_cold = int(len(users) * cold_start_ratio)
        cold_users = set(np.random.choice(users, n_cold, replace=False))
        train_data = [pair for pair in train_data if pair[0] not in cold_users]
        print(f"[Data] Cold-start users: {n_cold} users removed, remaining {len(train_data)} interactions")
    elif setting == 'cold_item':
        items = list(set([i for _, i in train_data]))
        n_cold = int(len(items) * cold_start_ratio)
        cold_items = set(np.random.choice(items, n_cold, replace=False))
        train_data = [pair for pair in train_data if pair[1] not in cold_items]
        print(f"[Data] Cold-start items: {n_cold} items removed, remaining {len(train_data)} interactions")
    else:
        raise ValueError(f"Unknown setting: {setting}")

    # 更新训练数据
    dp.train_data = train_data

    # 获取训练数据中出现的所有用户和物品
    train_users = set([u for u, _ in train_data])
    train_items = set([i for _, i in train_data])

    # 重新计算用户数和物品数（基于训练数据中的最大ID+1，但注意ID可能不连续）
    # 为了保持ID的连续性，我们重新映射ID（可选），但为了简化，我们只保留训练数据中出现的用户和物品，
    # 并更新 n_users, n_items 为这些集合的大小（如果原ID是连续的，则直接取max+1）
    # 这里采用最大ID+1的方式，但需要确保所有ID都在[0, max_id]范围内。
    # 由于原始数据中ID是连续的，max+1即可。
    dp.n_users = max(train_users) + 1 if train_users else 0
    dp.n_items = max(train_items) + 1 if train_items else 0
    print(f"[Data] Updated graph size - Users: {dp.n_users}, Items: {dp.n_items}")

    # 重建 train_dict
    dp.train_dict = dp._build_user_item_dict(train_data, as_set=True)

    # 对于非 full 协议，禁用 meta_val_loader（避免索引越界）
    if setting != 'full':
        dp.meta_val_loader = None
        print("[Data] Disabled meta_val_loader (will use train sampling for meta validation).")

    # 裁剪测试集：只保留训练数据中存在的用户和物品的交互
    if hasattr(dp, 'test_dict') and dp.test_dict is not None:
        # test_dict 是 dict: user -> list of items
        new_test_dict = {}
        for user, items in dp.test_dict.items():
            if user in train_users:
                filtered_items = [i for i in items if i in train_items]
                if filtered_items:
                    new_test_dict[user] = filtered_items
        dp.test_dict = new_test_dict
        print(f"[Data] Filtered test_dict: kept {len(new_test_dict)} users")

        total_test_interactions = sum(len(items) for items in new_test_dict.values())
        print(f"[Data] Filtered test set: {total_test_interactions} interactions")

        # 过滤辅助链接，只保留训练集中存在的用户（针对 Yelp 社交边）
        if hasattr(dp, 'aux_links') and dp.aux_links is not None and len(dp.aux_links) > 0:
            # 确保 aux_links 中的用户 ID 都在 [0, dp.n_users) 范围内
            mask = (dp.aux_links[:, 0] < dp.n_users) & (dp.aux_links[:, 1] < dp.n_users)
            dp.aux_links = dp.aux_links[mask]
            print(f"[Data] Filtered aux_links: kept {len(dp.aux_links)} edges")

    # 重新构建归一化邻接矩阵（图结构已变）
    dp.norm_adj = dp._build_norm_adj()

    # 重新创建训练 DataLoader
    dp.train_loader = DataLoader(
        RecDataset(train_data, dp.train_dict, dp.n_items),
        batch_size=dp.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    # 注意：meta_val_loader 保持不变（验证集不应随训练集改变）
    return dp


# ================= 主实验函数 =================
def run_improved_efficiency_analysis():
    print("=" * 60)
    print("🚀 Running IMPROVED RQ3: Sparse/Cold-start Protocols")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 基础配置
    base_config = {
        'dataset': 'yelp_processed_for_meta',
        'model_name': 'LightGCN',
        'lr': 0.001,
        'meta_lr': 0.0005,
        'wd': 1e-4,
        'n_layers': 2,
        'embed_dim': 64,
        'hvp_eps': 1e-4,
        'meta_warmup_epochs': 20,
        'meta_loss_weight': 0.1,
    }

    # IM-Net 增强配置（更大容量）
    imnet_config = base_config.copy()
    imnet_config['embed_dim'] = 128
    imnet_config['n_layers'] = 3
    imnet_config['mode'] = 'meta'

    # Single Task 基线配置（原始容量）
    single_config = base_config.copy()
    single_config['embed_dim'] = 64
    single_config['n_layers'] = 2
    single_config['mode'] = 'single'

    # 实验协议列表（不含 conflict，因其需修改 train_engine）
    settings = [
        ('full', 'Full Data'),
        ('few_10', '10% Data'),
        ('few_20', '20% Data'),
        ('few_50', '50% Data'),
        ('cold_user', 'Cold Users'),
        ('cold_item', 'Cold Items')
    ]

    max_epochs = 200
    all_results = []

    for setting_name, setting_desc in settings:
        print(f"\n{'=' * 60}")
        print(f">>> Protocol: {setting_desc}")
        print(f"{'=' * 60}")

        # 加载原始数据
        dp_orig = DataProcessor(base_config['dataset'], dataset_type='yelp', batch_size=2048)
        dp = prepare_experiment_data(dp_orig, setting=setting_name)

        # ---------- 1. IM-Net (Proactive, larger capacity) ----------
        print("\n[IM-Net] Training...")
        model_imnet = LightGCN(dp.n_users, dp.n_items,
                               imnet_config['embed_dim'],
                               imnet_config['n_layers']).to(device)
        trainer_imnet = IMNetTrainer(dp, model_imnet, imnet_config, device)

        cum_time_imnet = 0.0
        best_imnet = {'recall': 0.0, 'ndcg': 0.0, 'epoch': 0}

        for epoch in range(1, max_epochs + 1):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            loss = trainer_imnet.train_epoch(epoch, mode='meta')

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            epoch_t = time.perf_counter() - start
            cum_time_imnet += epoch_t

            recall, ndcg = trainer_imnet.evaluate(top_k=20)

            if ndcg > best_imnet['ndcg']:
                best_imnet['recall'] = recall
                best_imnet['ndcg'] = ndcg
                best_imnet['epoch'] = epoch

            all_results.append({
                'Protocol': setting_desc,
                'Method': 'IM-Net (Proactive)',
                'Epoch': epoch,
                'EpochTime': epoch_t,
                'CumulativeTime': cum_time_imnet,
                'Recall': recall,
                'NDCG': ndcg
            })

            if epoch % 20 == 0:
                print(f"  IM-Net Epoch {epoch:03d} | Time {epoch_t:.2f}s | Recall {recall:.5f} | NDCG {ndcg:.5f}")

        # ---------- 2. Single Task Baseline (original capacity) ----------
        print("\n[Single Task] Training...")
        model_single = LightGCN(dp.n_users, dp.n_items,
                                single_config['embed_dim'],
                                single_config['n_layers']).to(device)
        trainer_single = IMNetTrainer(dp, model_single, single_config, device)

        cum_time_single = 0.0
        best_single = {'recall': 0.0, 'ndcg': 0.0, 'epoch': 0}

        for epoch in range(1, max_epochs + 1):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            loss = trainer_single.train_epoch(epoch, mode='single')

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            epoch_t = time.perf_counter() - start
            cum_time_single += epoch_t

            recall, ndcg = trainer_single.evaluate(top_k=20)

            if ndcg > best_single['ndcg']:
                best_single['recall'] = recall
                best_single['ndcg'] = ndcg
                best_single['epoch'] = epoch

            all_results.append({
                'Protocol': setting_desc,
                'Method': 'Single Task (Reactive)',
                'Epoch': epoch,
                'EpochTime': epoch_t,
                'CumulativeTime': cum_time_single,
                'Recall': recall,
                'NDCG': ndcg
            })

            if epoch % 20 == 0:
                print(f"  Single Epoch {epoch:03d} | Time {epoch_t:.2f}s | Recall {recall:.5f} | NDCG {ndcg:.5f}")

        # 打印协议总结
        recall_gain = best_imnet['recall'] - best_single['recall']
        ndcg_gain = best_imnet['ndcg'] - best_single['ndcg']
        recall_rel = (recall_gain / (best_single['recall'] + 1e-8)) * 100
        ndcg_rel = (ndcg_gain / (best_single['ndcg'] + 1e-8)) * 100

        print(f"\n[Summary] {setting_desc}")
        print(
            f"  IM-Net   -> Best Recall: {best_imnet['recall']:.5f}, Best NDCG: {best_imnet['ndcg']:.5f} (epoch {best_imnet['epoch']})")
        print(
            f"  Single   -> Best Recall: {best_single['recall']:.5f}, Best NDCG: {best_single['ndcg']:.5f} (epoch {best_single['epoch']})")
        print(
            f"  Improvement: ΔRecall = {recall_gain:+.5f} ({recall_rel:+.2f}%), ΔNDCG = {ndcg_gain:+.5f} ({ndcg_rel:+.2f}%)")
        print("-" * 60)

    # 保存所有结果
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv('results/rq3_improved_protocols.csv', index=False)
    print("\n[INFO] Results saved to results/rq3_improved_protocols.csv")

    # 绘制对比图
    plot_improved_results(df)


def plot_improved_results(df):
    """绘制不同协议下 IM-Net vs Single Task 的最佳性能对比"""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # 提取每个协议的最佳性能（按 NDCG）
    best_df = df.groupby(['Protocol', 'Method']).apply(
        lambda x: x.loc[x['NDCG'].idxmax()]
    ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall 对比
    sns.barplot(data=best_df, x='Protocol', y='Recall', hue='Method', ax=axes[0],
                palette={'IM-Net (Proactive)': '#E53935', 'Single Task (Reactive)': '#90CAF9'})
    axes[0].set_title('Best Recall@20 Comparison', fontweight='bold')
    axes[0].set_ylabel('Recall')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=30)

    # NDCG 对比
    sns.barplot(data=best_df, x='Protocol', y='NDCG', hue='Method', ax=axes[1],
                palette={'IM-Net (Proactive)': '#E53935', 'Single Task (Reactive)': '#90CAF9'})
    axes[1].set_title('Best NDCG@20 Comparison', fontweight='bold')
    axes[1].set_ylabel('NDCG')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig('results/improved_protocols_comparison.png', dpi=300)
    plt.savefig('results/improved_protocols_comparison.pdf', bbox_inches='tight')
    print("[INFO] Figure saved to results/improved_protocols_comparison.png")
    plt.show()


if __name__ == '__main__':
    run_improved_efficiency_analysis()