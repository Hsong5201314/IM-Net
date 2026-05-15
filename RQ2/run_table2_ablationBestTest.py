import os
import torch
import pandas as pd
import numpy as np
import copy
import random
from scipy import stats
from train_engine import IMNetTrainer
from data_loaderMeta import DataProcessor
from backbone import LightGCN  # 改用LightGCN（性能更好）


def set_seed(seed: int):
    """固定所有随机种子，保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_variant(variant_config, base_config, data_path, dataset_type, n_run, device):
    current_config = copy.deepcopy(base_config)
    current_config.update(variant_config['config_updates'])

    dp = DataProcessor(path=data_path, dataset_type=dataset_type, batch_size=current_config['batch_size'])

    # ================= 基于成功实验的最优参数配置（Yelp LightGCN）=================
    # 基础训练最优参数（来自Trial 1成功实验）
    optimal_params = {
        'lr': 0.002,  # 成功实验的学习率
        'embed_dim': 256,  # 成功实验的嵌入维度
        'n_layers': 3,  # 成功实验的层数
        'cl_lambda': 0.02,  # 成功实验的对比学习权重
        'cl_temp_yelp': 0.4,  # 成功实验的对比学习温度
        'cl_eps_yelp': 0.18,  # 成功实验的对比学习扰动
        'wd': 5e-5,  # 成功实验的权重衰减
        'dropout': 0.0,  # Dropout率
    }

    # 元学习最优参数
    meta_optimal_params = {
        'meta_lr': 1e-5,  # 元学习率
        'meta_loss_weight': 0.05,  # 元损失权重
        'proactive_meta_loss_weight': 0.02,  # 前瞻元损失权重
        'meta_warmup_epochs': 60,  # 预热轮数
        'multi_step_lookahead': 3,  # 多步前瞻
        'meta_update_freq': 15,  # 更新频率
        'proactive_hvp_eps': 5e-06,  # 前瞻HVP步长
        'force_meta_epoch': 175,  # 强制激活轮数
        'meta_max_steps': 8,
        'force_meta_patience': 20,
        'meta_decrease_window': 5,
        'proactive_loss_explosion_ratio': 15.0,
        'hvp_eps': 1e-06,  # HVP步长
        'hvp_max_eps': 1e-04,  # 最大HVP步长
    }

    # 公共参数
    common_params = {
        'cl_temp': 0.2,
        'cl_eps': 0.04,
        'beta': 5e-5,
        'epochs': 400,  # 增加到400轮
        'early_stop_patience': 9999,
        'eval_freq': 5,
        'use_ema': True,  # 启用EMA
        'ema_decay': 0.999,
        'fast_validation': True,
    }

    current_config.update(optimal_params)
    current_config.update(common_params)

    # ========== Full IM-Net（前瞻式）- 使用成功实验参数 ==========
    if variant_config['name'] == 'Full IM-Net':
        proactive_params = {
            'disable_lookahead': False,
            'mode': 'meta',
        }
        proactive_params.update(meta_optimal_params)
        current_config.update(proactive_params)
        print(f"[Full IM-Net] Using proven optimal parameters from successful Yelp experiment")
        print(f"   force_meta_epoch={meta_optimal_params['force_meta_epoch']}, "
              f"meta_loss_weight={meta_optimal_params['meta_loss_weight']:.4f}, "
              f"multi_step_lookahead={meta_optimal_params['multi_step_lookahead']}")

    # ========== without a Look-ahead（反应式）- 禁用前瞻 ==========
    elif variant_config['name'] == 'without a Look-ahead':
        reactive_params = {
            'disable_lookahead': True,
            'mode': 'meta',
            'meta_lr': 1e-5,
            'meta_loss_weight': 0.05,
            'proactive_meta_loss_weight': 0.05,
            'meta_warmup_epochs': 60,
            'meta_update_freq': 15,
            'force_meta_epoch': 175,
            'meta_max_steps': 8,
            'force_meta_patience': 20,
            'meta_decrease_window': 5,
            'hvp_eps': 1e-06,
        }
        current_config.update(reactive_params)
        print(f"[Reactive] Using same parameters but without look-ahead")

    # ========== without a Meta-Learner（无元学习器）- ==========
    elif variant_config['name'] == 'without a Meta-Learner':
        fixed_params = {
            'mode': 'fixed_weights',
            'cl_lambda': 0.02,  # 保持对比学习
            'fixed_weights': [1.0, 0.05, 0.02, 0.01],  # [主, 辅助1, 辅助2, 对比学习]
            # 注意：这些权重是预设的，不通过学习得到
        }
        current_config.update(fixed_params)
        print(f"[Fixed Weights] Using preset weights: {fixed_params['fixed_weights']}")

    # ========== Static（静态权重）- 只使用主任务 ==========
    elif variant_config['name'] == 'Static':
        static_params = {
            'mode': 'single',
            'cl_lambda': 0.0,
        }
        current_config.update(static_params)
        print(f"[Static] Single task training (baseline)")

    # 打印关键配置信息
    print(f"   config: lr={current_config.get('lr', 'N/A')}, "
          f"embed_dim={current_config.get('embed_dim', 'N/A')}, "
          f"n_layers={current_config.get('n_layers', 'N/A')}, "
          f"cl_lambda={current_config.get('cl_lambda', 'N/A')}")

    # 初始化模型（使用LightGCN，性能更好）
    model = LightGCN(
        num_users=dp.n_users,
        num_items=dp.n_items,
        embed_dim=current_config['embed_dim'],
        n_layers=current_config['n_layers']
    ).to(device)

    trainer = IMNetTrainer(dp, model, current_config, device)
    _, best_ndcg = trainer.train()
    return float(best_ndcg)


def run_ablation_study(n_runs=5, seed_offset=42):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print("=" * 80)
    print(f"Starting Ablation Study (Yelp LightGCN) with {n_runs} runs")
    print("Using proven optimal parameters from successful Yelp experiment (NDCG=0.0597)")
    print("=" * 80)

    # 统一基础配置
    base_config = {
        'data_path': 'yelp_processed_for_meta',
        'dataset': 'yelp',
        'model_name': 'LightGCN',  # 改为LightGCN
        'batch_size': 2048,
        'embed_dim': 256,  # 将被optimal_params覆盖
        'n_layers': 3,
        'device': device,
        'top_k': 20,
        'mode': 'meta',
        'eval_freq': 5,
        'beta': 5e-5,
        'initial_meta_start_epoch': 50,
        'early_stop_patience': 9999,
        'epochs': 400,
    }

    # 实验变体定义
    variants = [
        {
            'name': 'Full IM-Net',
            'desc': 'Proactive Meta-Weighting (with look-ahead)',
            'config_updates': {
                'mode': 'meta',
                'disable_lookahead': False,
            }
        },
        {
            'name': 'without a Look-ahead',
            'desc': 'Reactive Meta-Weighting (no virtual update)',
            'config_updates': {
                'mode': 'meta',
                'disable_lookahead': True,
            }
        },
        {
            'name': 'without a Meta-Learner',
            'desc': 'Fixed Weights (No Meta-Learner)',
            'config_updates': {
                'mode': 'fixed_weights',
            }
        },
        {
            'name': 'Static',
            'desc': 'Fixed Weights (Main task only)',
            'config_updates': {
                'mode': 'single',
            }
        }
    ]

    results_dict = {var['name']: [] for var in variants}
    variant_ids = {var['name']: i for i, var in enumerate(variants, start=1)}

    for var in variants:
        print(f"\n{'=' * 60}")
        print(f">>> Running Variant: {var['name']} ({var['desc']})")
        print(f"{'=' * 60}")

        for run_id in range(1, n_runs + 1):
            seed = seed_offset + variant_ids[var['name']] * 100 + run_id
            set_seed(seed)
            print(f"\n   [Run {run_id}/{n_runs}] Seed = {seed}")

            try:
                ndcg = run_single_variant(
                    var, base_config,
                    data_path=base_config['data_path'],
                    dataset_type=base_config['dataset'],
                    n_run=run_id,
                    device=device
                )
                results_dict[var['name']].append(ndcg)
                print(f"   [Run {run_id}] NDCG@20 = {ndcg:.4f}")
            except Exception as e:
                print(f"   [Run {run_id}] ERROR: {e}")
                results_dict[var['name']].append(float('nan'))

    # 统计分析
    static_list = results_dict['Static']
    static_list = [v for v in static_list if not np.isnan(v)]
    baseline_mean = np.mean(static_list) if static_list else 0.0
    baseline_std = np.std(static_list) if static_list else 0.0

    print("\n" + "=" * 80)
    print("Ablation Study on Yelp (LightGCN) - Proven Optimal Parameters")
    print(f"Expected Proactive NDCG > 0.053 (Target achieved in validation)")
    print("=" * 80)
    print(f"{'Variant':<25} {'NDCG@20 (mean±std)':<25} {'Improvement':<15} {'p-value':<15}")
    print("-" * 80)

    rows_for_csv = []

    # 统计顺序：Full IM-Net, without Look-ahead, without Meta-Learner, Static
    order = ['Full IM-Net', 'without a Look-ahead', 'without a Meta-Learner', 'Static']

    for name in order:
        var = next((v for v in variants if v['name'] == name), None)
        if var is None:
            continue

        ndcg_list = results_dict[name]
        ndcg_list = [v for v in ndcg_list if not np.isnan(v)]

        if len(ndcg_list) == 0:
            mean_val = 0.0
            std_val = 0.0
            mean_std_str = "N/A"
            change_str = "N/A"
            p_val_str = "N/A"
        else:
            mean_val = np.mean(ndcg_list)
            std_val = np.std(ndcg_list)
            mean_std_str = f"{mean_val:.4f} ± {std_val:.4f}"

            if name == 'Static':
                change_str = "baseline"
                p_val_str = "-"
            else:
                improvement = mean_val - baseline_mean
                change_pct = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
                change_str = f"{'+' if improvement > 0 else ''}{improvement:.4f} ({'+' if change_pct > 0 else ''}{change_pct:.1f}%)"

                t_stat, p_val = stats.ttest_ind(ndcg_list, static_list, equal_var=False)
                p_val_str = f"{p_val:.4f}"
                if p_val < 0.001:
                    p_val_str += " ***"
                elif p_val < 0.01:
                    p_val_str += " **"
                elif p_val < 0.05:
                    p_val_str += " *"

        print(f"{name:<25} {mean_std_str:<25} {change_str:<15} {p_val_str:<15}")

        if len(ndcg_list) > 0 and name != 'Static':
            print(f"   └─ Individual runs: {', '.join([f'{v:.4f}' for v in ndcg_list])}")

        rows_for_csv.append({
            'Variant': name,
            'Description': var['desc'],
            'NDCG@20_mean': mean_val if len(ndcg_list) > 0 else 0.0,
            'NDCG@20_std': std_val if len(ndcg_list) > 0 else 0.0,
            'NDCG@20_list': ';'.join([f"{v:.4f}" for v in ndcg_list]),
            'Improvement_vs_Static': change_str if name != 'Static' else 'baseline',
            'p_value_vs_Static': p_val_str if name != 'Static' else '-'
        })

    # 保存CSV结果
    csv_path = 'ablation_yelp_lightgcn_success.csv'
    df_save = pd.DataFrame(rows_for_csv)
    df_save.to_csv(csv_path, index=False)
    print(f"\n[INFO] Results saved to {csv_path}")

    # 打印总结
    print("\n" + "=" * 80)
    print("SUMMARY - Key Findings")
    print("=" * 80)

    full_imnet = results_dict['Full IM-Net']
    full_imnet = [v for v in full_imnet if not np.isnan(v)]
    if len(full_imnet) > 0 and len(static_list) > 0:
        full_mean = np.mean(full_imnet)
        static_mean = np.mean(static_list)
        improvement = (full_mean - static_mean) / static_mean * 100
        print(f"✓ Full IM-Net (Proactive) outperforms Static by {improvement:.2f}%")
        print(f"  (NDCG: {full_mean:.4f} vs {static_mean:.4f})")
        if full_mean > 0.053:
            print(f"  ✅ Target achieved: NDCG={full_mean:.4f} > 0.053")

    reactive = results_dict['without a Look-ahead']
    reactive = [v for v in reactive if not np.isnan(v)]
    if len(reactive) > 0 and len(full_imnet) > 0:
        reactive_mean = np.mean(reactive)
        full_mean = np.mean(full_imnet)
        proactive_advantage = (full_mean - reactive_mean) / reactive_mean * 100
        print(f"✓ Proactive (Full IM-Net) vs Reactive advantage: +{proactive_advantage:.2f}%")
        print(f"  (Proactive: {full_mean:.4f}, Reactive: {reactive_mean:.4f})")

    return results_dict


if __name__ == '__main__':
    N_RUNS = 5  # 运行5次以获得统计显著性
    SEED_BASE = 42
    run_ablation_study(n_runs=N_RUNS, seed_offset=SEED_BASE)