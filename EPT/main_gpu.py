import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
import random
from tqdm import tqdm

# --- Module Imports ---
from data_loaderMeta import DataProcessor
from backbone import LightGCN, NCF, SimGCL, HINE
from imnet import IMNet
from train_engine import IMNetTrainer

# ================= 1. Expert-level Differentiated Configurations =================
PAPER_SETTINGS = {
    'yelp': {
        'batch_size': 2048,
        'lr': 0.001,
        'meta_lr': 0.0001,
        'n_layers': 3,
        'wd': 1e-4,
        'dropout': 0.0,
        'beta': 5e-5,
        'tau': 0.2,
        'embed_dim': 64,
        'cl_lambda': 0.1,
        'cl_eps': 0.15,
        'cl_temp': 0.2,
        'initial_meta_start_epoch': 20,
        'meta_decrease_window': 3,
        'eval_freq': 1,
        'meta_target_scale': 0.2,
        'meta_loss_weight': 1.0,
    },
    'amazon': {
        'batch_size': 2048,
        'lr': 0.001,
        'meta_lr': 2e-5,
        'n_layers': 3,
        'wd': 1e-4,
        'dropout': 0.0,
        'beta': 5e-5,
        'tau': 0.2,
        'embed_dim': 128,
        'cl_lambda': 0.1,
        'cl_eps': 0.15,
        'cl_temp': 0.2,
        'initial_meta_start_epoch': 20,
        'meta_decrease_window': 3,
        'eval_freq': 5,
        'meta_target_scale': 0.2,
    }
}


@torch.no_grad()
def get_metrics(model, model_name, norm_adj, train_dict, test_dict, device, top_k=20):
    """GPU加速的评估函数（保持不变）"""
    model.eval()
    if model_name == 'NCF':
        all_u_emb, all_i_emb = model.get_all_embeddings()
    else:
        all_u_emb, all_i_emb = model.get_all_embeddings(norm_adj.to(device))
    test_users = list(test_dict.keys())
    batch_size = 4096
    total_recall = 0.0
    total_ndcg = 0.0
    hit_users = 0
    for i in range(0, len(test_users), batch_size):
        batch_users = test_users[i:i + batch_size]
        u_emb = all_u_emb[batch_users]
        scores = torch.matmul(u_emb, all_i_emb.T)
        for idx, u in enumerate(batch_users):
            train_items = list(train_dict.get(u, []))
            if len(train_items) > 0:
                scores[idx, train_items] = -float('inf')
        _, top_items = torch.topk(scores, top_k, dim=1)
        top_items = top_items.cpu().numpy()
        for idx, u in enumerate(batch_users):
            test_items = list(test_dict.get(u, []))
            if len(test_items) == 0:
                continue
            hit_users += 1
            hits = np.isin(top_items[idx], test_items)
            total_recall += np.sum(hits) / len(test_items)
            dcg = np.sum(hits / np.log2(np.arange(2, top_k + 2)))
            idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(test_items), top_k) + 2)))
            total_ndcg += dcg / idcg
    if hit_users == 0:
        return 0.0, 0.0
    return total_recall / hit_users, total_ndcg / hit_users


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="IM-Net Optimized for Amazon/Yelp")
    parser.add_argument('--dataset', type=str, default='amazon', help='yelp / amazon')
    parser.add_argument('--model_name', type=str, default='LightGCN', help='LightGCN/NCF/SimGCL/HINE')
    parser.add_argument('--mode', type=str, default='meta', help='single/baseline/moo/meta')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./yelp_processed_for_meta')
    parser.add_argument('--cl_lambda', type=float, default=0.2, help='Weight for Contrastive Learning Loss')
    parser.add_argument('--cl_temp', type=float, default=0.2, help='Temperature for InfoNCE')
    parser.add_argument('--wd', type=float, default=None, help='Weight Decay')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last best model')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed for reproducible experiments')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--phase', type=str, default='full_proactive',
                        choices=['reactive', 'proactive', 'dual', 'full_proactive'],
                        help='Training phase: reactive, proactive, dual, or full_proactive')
    parser.add_argument('--simulation', action='store_true',
                        help='Enable simulation mode with severe gradient conflict (ρ≈-1)')
    parser.add_argument('--imnet_input_dim', type=int, default=4,
                        help='Input dimension of IMNet (5 for simulation mode)')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate for NCF/LightGCN (if supported)')
    parser.add_argument('--embed_dim', type=int, default=None, help='Embedding dimension for user/item')

    # 优化参数
    parser.add_argument('--enable_ema', action='store_true', help='Enable EMA model evaluation')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--enable_warmup', action='store_true', help='Enable learning rate warmup')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--fast_validation', action='store_true', default=True, help='Enable fast validation')
    parser.add_argument('--multi_step', type=int, default=None, help='Multi-step lookahead steps')
    parser.add_argument('--simulated', action='store_true',
                        help='Use simulated data for quick phase transition verification')
    parser.add_argument('--meta_lr', type=float, default=5.575453980775358e-08,
                        help='Learning rate for the meta-optimizer (IMNet)')
    # 在 parser.add_argument('--meta_lr', ...) 之后添加
    parser.add_argument('--inner_lr', type=float, help='Virtual update inner learning rate')
    parser.add_argument('--lambda_quad', type=float, help='Quadratic penalty coefficient')
    parser.add_argument('--lambda_interference', type=float, help='Interference penalty coefficient')
    parser.add_argument('--lambda_sparsity', type=float, default=None,
                        help='Sparsity regularization coefficient for w_aux (prevents permanent zero)')
    parser.add_argument('--lambda_conflict', type=float, default=None, help='Conflict penalty coefficient')

    parser.add_argument('--meta_update_freq', type=int, help='IMNet update frequency')
    parser.add_argument('--peak_lambda', type=float, default=None, help='Peak grad_reverse_lambda')
    parser.add_argument('--decay_start', type=int, default=None, help='Decay start epoch')
    parser.add_argument('--decay_end', type=int, default=None, help='Decay end epoch')
    parser.add_argument('--min_lambda', type=float, default=None, help='Minimum grad_reverse_lambda')
    parser.add_argument('--phase_switch_epoch', type=int, help='Phase switch epoch')
    parser.add_argument('--conflict_scale', type=float, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load base config
    config = PAPER_SETTINGS[args.dataset]
    # 用户显式指定的超参数优先（必须在任何覆盖之前）
    user_overrides = {}
    # 添加超参数覆盖
    if args.inner_lr is not None:
        config['inner_lr'] = args.inner_lr
    if args.lambda_quad is not None:
        config['lambda_quad'] = args.lambda_quad
    if args.lambda_interference is not None:
        config['lambda_interference'] = args.lambda_interference
    if args.lambda_sparsity is not None:
        config['lambda_sparsity'] = args.lambda_sparsity

    if args.lambda_conflict is not None:
        config['lambda_conflict'] = args.lambda_conflict

    if args.meta_update_freq is not None:
        config['meta_update_freq'] = args.meta_update_freq
    if args.peak_lambda is not None:
        config['peak_lambda'] = args.peak_lambda
    if args.decay_start is not None:
        config['decay_start'] = args.decay_start
    if args.decay_end is not None:
        config['decay_end'] = args.decay_end
    if args.min_lambda is not None:
        config['min_lambda'] = args.min_lambda
    if args.phase_switch_epoch is not None:
        config['phase_switch_epoch'] = args.phase_switch_epoch
    if args.embed_dim is not None:
        config['embed_dim'] = args.embed_dim
    if args.meta_lr != 5.575453980775358e-08:  # 如果用户修改了默认值
        user_overrides['meta_lr'] = args.meta_lr
    if args.wd is not None:
        user_overrides['wd'] = args.wd
    if args.dropout is not None:
        user_overrides['dropout'] = args.dropout
    if args.lr is not None:
        user_overrides['lr'] = args.lr
    if args.batch_size is not None:
        user_overrides['batch_size'] = args.batch_size
    config.update(user_overrides)
    config['model_name'] = args.model_name
    config['dataset'] = args.dataset
    config['mode'] = args.mode
    config['cl_lambda'] = args.cl_lambda
    config['cl_temp'] = args.cl_temp
    config['epochs'] = args.epochs
    config['meta_lr'] = args.meta_lr
    if args.wd is not None:
        config['wd'] = args.wd

    # ================= 仿真模式专属配置 =================
    if args.simulation:
        print("\n" + "=" * 60)
        print("[SIMULATION MODE] Enabling severe gradient conflict (ρ ≈ -1)")
        print("  - IM-Net input dimension = 5 (losses + interference energy)")
        print("  - Hessian spectral radius logged every 10 batches")
        print("  - Interference energy and weight trajectories recorded")
        print("=" * 60 + "\n")
        config.update({
            'simulation_mode': True,
            'imnet_input_dim': 5,
            'hessian_freq': 10,
            'rho_target': -0.99,
            'conflict_scale': 2.0,
            'lr': 0.001,
            'wd': 0.002,
            'dropout': 0.2,
            'disable_lookahead': False,  # 开启前瞻式（关键！）
            'lookahead_switch_epoch': None,
            'multi_step_lookahead': 3,  # 2步前瞻
            'meta_lr': 5e-4,  # 元学习率（将被后续 Yelp 覆盖）
            'proactive_meta_loss_weight': 0.01,  # 降低元损失权重
            'lambda_spectral': 0.1,  # 提高谱半径惩罚（将被覆盖）
            'lambda_interference': 0.0,  # 完全禁用干涉惩罚
            'interference_threshold': 0.001,
            'lambda_conflict': 20.0,
            'lambda_sparsity': 0.5,
            'meta_warmup_epochs': 3,  # 极短预热（将被覆盖）
            'force_meta_epoch': 5,
            'conflict_warmup_ratio': 0.2,
            'inner_lr': 0.05,  # 临时值，将被覆盖
            'meta_update_freq': 5,  # 临时值，将被覆盖
            'lambda_quad': 0.0,  # 关闭二次惩罚（将被覆盖）
            'eval_freq': 1,
            'cl_lambda': 0.0,
            'fixed_weights': [1.0, 1.0, 0.0, 0.0],  # Static 模式使用
            'min_main_weight': 0.3,  # 临时值，将被覆盖
        })
        if config['epochs'] < 100:
            config['epochs'] = 100
        os.makedirs('simulation_logs', exist_ok=True)

        # 【新增】仿真模式下自动启用 EMA（如果用户未显式禁用）
        if not args.enable_ema:
            args.enable_ema = True
            print("[SIMULATION] Auto-enabled EMA for stable evaluation.")

    # 在仿真模式专属配置之后（或全局 config 更新处），加入：
    if args.peak_lambda is not None:
        config['peak_lambda'] = args.peak_lambda
    if args.decay_start is not None:
        config['decay_start'] = args.decay_start
    if args.decay_end is not None:
        config['decay_end'] = args.decay_end
    if args.min_lambda is not None:
        config['min_lambda'] = args.min_lambda
    if args.phase_switch_epoch is not None:
        config['phase_switch_epoch'] = args.phase_switch_epoch

    # ========== 原有超参数覆盖逻辑（保持不变） ==========
    if args.mode == 'meta' and args.model_name == 'LightGCN':
        if args.dataset == 'amazon':
            config['embed_dim'] = 256
            config['n_layers'] = 4
            config['lr'] = 0.0006
            config['wd'] = 0.001
            config['cl_lambda'] = 0.00
            config['cl_eps_amazon'] = 0.1
            config['cl_temp_amazon'] = 0.2
            if args.phase == 'full_proactive':
                config['disable_lookahead'] = False
                config['lookahead_switch_epoch'] = None
                config['multi_step_lookahead'] = 3
                config['proactive_hvp_eps'] = 5e-06
                config['proactive_meta_loss_weight'] = 0.01
            elif args.phase == 'proactive':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = 150
                config['multi_step_lookahead'] = 3
            else:
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = None
            config['meta_lr'] = 1e-5
            config['meta_loss_weight'] = 0.01
            config['proactive_meta_loss_weight'] = 0.01
            config['hvp_eps'] = 1e-4
            config['hvp_max_eps'] = 1e-5
            config['proactive_hvp_eps'] = 1e-4
            config['proactive_hvp_max_eps'] = 1e-4
            config['force_meta_epoch'] = 30
            config['meta_warmup_epochs'] = 40
            config['meta_max_steps'] = 6
            config['meta_update_freq'] = 10
            config['proactive_loss_explosion_ratio'] = 5.0
            if args.enable_ema:
                config['use_ema'] = True
                config['ema_decay'] = 0.999
            if args.grad_accum > 1:
                config['grad_accumulation_steps'] = args.grad_accum
            if args.enable_warmup:
                config['use_warmup'] = True
                config['warmup_epochs'] = args.warmup_epochs
            if config['epochs'] < 100:
                config['epochs'] = 100

            # ===== 新增：amazon保留仿真模式设置 =====
            if args.simulation:
                config['simulation_mode'] = True
                config['imnet_input_dim'] = 5
                config['hessian_freq'] = 10  # 从 1 → 10（每10个batch算一次）
                config['rho_target'] = -0.999
                config['meta_lr'] = 5e-5
                config['proactive_meta_loss_weight'] = 0.05
                config['disable_lookahead'] = False
                config['lookahead_switch_epoch'] = None
                config['multi_step_lookahead'] = 3
                config['eval_freq'] = 5
                config['cl_lambda'] = 0.0
                config['conflict_scale'] = 50.0
                config['meta_update_freq'] = 10
                config['inner_lr'] = 0.1
                config['lambda_conflict'] = 20.0
                config['lambda_sparsity'] = 0.5
                print("[SIMULATION] Force-kept simulation mode in Amazon config.")
        else:  # Yelp
            config['embed_dim'] = 128  # 临时因仿真需要从256修改为128
            config['n_layers'] = 3
            config['lr'] = 0.0037977679442478553
            config['cl_lambda'] = 0.16073441537982291
            config['wd'] = 1e-4
            config['meta_lr'] = 5.575453980775358e-08
            config['meta_loss_weight'] = 0.044842831927518936
            config['proactive_meta_loss_weight'] = 0.027888427611951233
            _epochs = config.get('epochs', 120)
            config['meta_warmup_epochs'] = max(5, int(_epochs * 0.15))
            # simulation模式强制元学习尽早激活
            config['force_meta_epoch'] = max(5, int(_epochs * 0.10))

            config['multi_step_lookahead'] = 3
            if args.meta_update_freq is None:
                config['meta_update_freq'] = 1
            config['proactive_hvp_eps'] = 2.1387290754148906e-07
            if not args.simulation:
                config['force_meta_epoch'] = 107
            config['reg_val_yelp'] = 5.3167142741246e-05
            config['cl_temp_yelp'] = 0.42720590636899725
            config['cl_eps_yelp'] = 0.17910958748845152
            config['hvp_eps'] = config['proactive_hvp_eps']
            config['proactive_hvp_max_eps'] = config['proactive_hvp_eps']
            config['hvp_max_eps'] = config['proactive_hvp_eps']
            config['grad_clip_norm'] = 1.0
            config['proactive_loss_explosion_ratio'] = 12.123391106782762
            if not args.simulation:
                config['loss_explosion_threshold'] = 10.0  # 如果效果不好改回2.0
            config['force_meta_patience'] = 12
            config['meta_decrease_window'] = 4
            config['meta_max_steps'] = 5
            config['fixed_meta_val_batch'] = False
            config['adaptive_epsilon'] = True
            config['disable_lookahead'] = False
            config['lookahead_switch_epoch'] = None
            if args.phase == 'reactive':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = None
            elif args.phase == 'proactive':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = 200
                config['proactive_hvp_eps'] = 1e-6
                config['force_meta_epoch'] = 200
            elif args.phase == 'dual':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = 200
                config['proactive_hvp_eps'] = 1e-6
                config['force_meta_epoch'] = 200
                config['dual_aux'] = True
            elif args.phase == 'full_proactive':
                config['disable_lookahead'] = False
                config['lookahead_switch_epoch'] = None
            if config['epochs'] < 100:
                config['epochs'] = 100

            # ===== 新增：Yelp保留仿真模式设置2（优化后版本） =====
            if args.simulation:
                config['simulation_mode'] = True
                config['imnet_input_dim'] = 5
                config['hessian_freq'] = 10
                config['rho_target'] = -0.99
                config['cl_lambda'] = 0.0
                # 核心优化参数
                config['meta_lr'] = 0.01                     # 元学习率大幅提高
                config['meta_update_freq'] = 1               # 每个batch更新元学习器
                config['lr'] = 0.001
                config['virtual_noise_scale'] = 0.2          # 探索噪声
                config['loss_explosion_threshold'] = 50.0
                config['lambda_interference'] = 0.0          # 完全禁用干涉惩罚
                config['lambda_spectral'] = 0.01             # 谱半径惩罚系数
                config['lambda_conflict'] = 20.0
                config['lambda_sparsity'] = 1.0              # 增强稀疏惩罚，驱动w_aux下降
                config['lambda_quad'] = 1.0                  # 启用二次惩罚
                config['fixed_weights'] = [1.0, 0.25, 0.25, 0.25]
                config['meta_warmup_epochs'] = 2
                config['eval_freq'] = 1
                config['epochs'] = max(config.get('epochs', 100), 100)
                config['conflict_scale'] = 1.0
                config['inner_lr'] = 0.5                     # 极大虚拟步长，使虚拟更新效果明显
                config['min_main_weight'] = 0.7              # 强制主任务权重不低于0.7
                config['peak_lambda'] = 1.0                  # Phase II 强冲突
                print("[SIMULATION] LightGCN simulation config optimized for paper (Phase Transition, weight suppression).")

    # 覆盖命令行参数
    if args.multi_step is not None:
        config['multi_step_lookahead'] = args.multi_step
    if args.enable_ema:
        config['use_ema'] = True
        config['ema_decay'] = 0.999
    if args.grad_accum > 1:
        config['grad_accumulation_steps'] = args.grad_accum
    if args.enable_warmup:
        config['warmup_epochs'] = args.warmup_epochs
        config['use_warmup'] = True
    if args.lr is not None:
        config['lr'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    print(f"[Expert Recommendation] Training {args.model_name} on {args.dataset} in {args.mode} mode.")
    print(f"Configurations: {config}")

    # ================= 数据加载 =================
    # ================= 数据加载 =================
    if args.simulated:
        from simulated_data import SimulatedDataProcessor
        sim_config = {
            'n_users': 2000,  # 可调小以快速测试
            'n_items': 4000,
            'batch_size': config['batch_size'],
            'interactions_per_user': 20,
            'test_ratio': 0.2,
        }
        dp = SimulatedDataProcessor(sim_config)
        # 模拟数据强制使用仿真冲突模式
        args.simulation = True  # 确保冲突损失被启用
    else:
        data_config = {
            'data_path': args.data_path,
            'dataset': args.dataset,
            'batch_size': config['batch_size'],
        }
        if args.phase == 'dual':
            data_config['dual_aux'] = True
        if args.simulation:
            data_config['simulation_conflict'] = True  # 启用冲突辅助损失
        dp = DataProcessor(data_config)

    # ================= 模型初始化 =================
    print(f"[INFO] Initializing Backbone Model: {args.model_name}...")

    embed_dim = config.get('embed_dim', 64)
    if args.model_name == 'LightGCN':
        model = LightGCN(
            num_users=dp.n_users, num_items=dp.n_items,
            embed_dim=embed_dim, n_layers=config['n_layers']
        ).to(args.device)
    elif args.model_name == 'NCF':
        model = NCF(
            num_users=dp.n_users, num_items=dp.n_items,
            embed_dim=embed_dim, n_layers=config['n_layers']
        ).to(args.device)
    elif args.model_name == 'SimGCL':
        config['lr'] = 0.02
        print(f"[Special Patch] SimGCL detected: Boosting Learning Rate to {config['lr']}")
        model = SimGCL(
            num_users=dp.n_users, num_items=dp.n_items,
            embed_dim=embed_dim, n_layers=config['n_layers']
        ).to(args.device)
    elif args.model_name == 'HINE':
        model = HINE(
            num_users=dp.n_users, num_items=dp.n_items,
            embed_dim=embed_dim, n_layers=config['n_layers']
        ).to(args.device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # ================= 创建训练器 =================
    print(f"[INFO] Initializing IMNetTrainer...")
    trainer = IMNetTrainer(dp, model, config, args.device)

    # ================= 仿真模式：无需额外替换，IM‑Net 已在 Trainer 中正确初始化 =================
    if args.simulation:
        print("[SIMULATION] IM‑Net already initialized with input_dim=5 in IMNetTrainer.")
        # 确保仿真指标容器存在（若尚未创建则补充，一般已存在）
        if not hasattr(trainer, 'simulation_metrics'):
            trainer.simulation_metrics = {
                'epoch': [], 'batch': [], 'interference_energy': [],
                'w_main': [], 'w_aux': [], 'spectral_radius': [], 'grad_cos_sim': []
            }
        # 同步 Hessian 计算频率（如果需要）
        if 'hessian_freq' in config:
            trainer.hessian_freq = config['hessian_freq']

    # ================= 训练循环 =================
    best_recall = 0.0
    best_ndcg = 0.0
    best_epoch = 0
    history = []
    start_epoch = 1
    eval_freq = config.get('eval_freq', 1)

    if args.resume:
        checkpoint_path = f"best_{args.dataset}_{args.model_name}_{args.mode}.pth"
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))

    print("[INFO] Start Training...")
    max_epochs = config['epochs']
    for epoch in range(start_epoch, max_epochs + 1):
        start_time = time.time()
        loss = trainer.train_epoch(epoch=epoch, mode=args.mode)

        if epoch % eval_freq == 0 or epoch == 1:
            if args.enable_ema and epoch > config.get('warmup_epochs', 0):
                recall, ndcg = trainer.evaluate_with_ema(top_k=20)
            else:
                recall, ndcg = trainer.evaluate(top_k=20)
            if hasattr(trainer, 'update_lr'):
                trainer.update_lr(ndcg, epoch)
        else:
            recall, ndcg = 0.0, 0.0

        if ndcg > best_ndcg:
            best_recall = recall
            best_ndcg = ndcg
            best_epoch = epoch
            torch.save(model.state_dict(), f"best_{args.dataset}_{args.model_name}_{args.mode}.pth")
            if args.enable_ema and hasattr(trainer, 'ema_model'):
                torch.save(trainer.ema_model.state_dict(), f"best_ema_{args.dataset}_{args.model_name}_{args.mode}.pth")
            print(f"[Best Updated] Epoch {epoch} found new best NDCG: {best_ndcg:.4f}")

        epoch_time = time.time() - start_time
        if recall > 0:
            eval_info = f" | R@20: {recall:.4f} | N@20: {ndcg:.4f}"
            print(
                f"Epoch {epoch:03d} | Loss: {loss:.4f}{eval_info} | Best N@20: {best_ndcg:.4f} | Time: {epoch_time:.1f}s")
        else:
            eval_info = " | [Skipped Eval]"
            print(
                f"Epoch {epoch:03d} | Loss: {loss:.4f}{eval_info} | Best N@20: {best_ndcg:.4f} | Time: {epoch_time:.1f}s")

        history.append([epoch, recall, ndcg, loss])

        # ================= 仿真模式：保存指标 =================
        if args.simulation and hasattr(trainer, 'simulation_metrics'):
            # 将每个 batch 记录的数据保存到 CSV
            if len(trainer.simulation_metrics['epoch']) > 0:
                df_sim = pd.DataFrame(trainer.simulation_metrics)
                df_sim.to_csv(f"simulation_logs/sim_metrics_{args.dataset}_{args.mode}.csv", index=False)

    # 结果保存
    res_df = pd.DataFrame(history, columns=['epoch', 'recall', 'ndcg', 'loss'])
    res_df.to_csv(f"results_{args.dataset}_{args.model_name}_{args.mode}.csv", index=False)
    print(f"[FINAL RESULTS] Best Recall @ 20: {best_recall:.4f}, Best NDCG @ 20: {best_ndcg:.4f} at Epoch {best_epoch}")
    print(
        f"=== [FINAL_RESULT] Dataset: {args.dataset} | Model: {args.model_name} | Mode: {args.mode} | Seed: {args.seed} | Best Recall@20: {best_recall:.4f} | Best NDCG@20: {best_ndcg:.4f} ===")


if __name__ == '__main__':
    main()