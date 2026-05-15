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

# --- Module Imports (ensure these filenames match your local files) ---
from data_loaderMeta import DataProcessor
from backbone import LightGCN, NCF, SimGCL, HINE
from imnet import IMNet
from train_engine import IMNetTrainer

# ================= 1. Expert-level Differentiated Configurations =================
# Senior Researcher: Amazon must use large Batch and strong regularization; Yelp stays as is to stabilize metrics.
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


# ================= NEW: GPU-Accelerated Evaluation Function =================
@torch.no_grad()
def get_metrics(model, model_name, norm_adj, train_dict, test_dict, device, top_k=20):
    model.eval()
    # ... (function content remains unchanged) ...
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
    if hit_users == 0: return 0.0, 0.0
    return total_recall / hit_users, total_ndcg / hit_users


# ========================================================

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
                        help='Training phase: reactive, proactive, dual, or full_proactive (always lookahead)')
    parser.add_argument('--simulation', action='store_true',
                        help='Enable simulation mode with conflicting auxiliary task')

    # ========== 新增优化参数（默认关闭，不影响原有运行） ==========
    parser.add_argument('--enable_ema', action='store_true',
                        help='Enable EMA (Exponential Moving Average) model evaluation')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--enable_warmup', action='store_true',
                        help='Enable learning rate warmup')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs (default: 10)')

    parser.add_argument('--fast_validation', action='store_true', default=True, help='Enable fast validation')
    parser.add_argument('--multi_step', type=int, default=None,
                        help='Multi-step lookahead (default: 1 for reactive, 3 for proactive)')
    parser.add_argument('--meta_loss_weight', type=float, default=None, help='Override meta_loss_weight')
    parser.add_argument('--force_meta_epoch', type=int, default=None, help='Override force_meta_epoch')
    parser.add_argument('--meta_update_freq', type=int, default=None, help='Override meta_update_freq')
    parser.add_argument('--neg_sample_ratio', type=int, default=None,
                        help='Negative sampling ratio (if data loader supports)')
    parser.add_argument('--curvature_reg_weight', type=float, default=0.0,
                        help='Weight for curvature regularization (default 0 = disabled)')
    parser.add_argument('--curvature_reg_threshold', type=float, default=0.5,
                        help='Threshold for curvature penalty (positive curvature above this value is penalized)')

    args = parser.parse_args()

    set_seed(2026)

    # Load base config
    config = PAPER_SETTINGS[args.dataset]
    config['model_name'] = args.model_name
    config['dataset'] = args.dataset
    config['mode'] = args.mode

    # Override with command line arguments
    config['cl_lambda'] = args.cl_lambda
    config['cl_temp'] = args.cl_temp
    config['epochs'] = args.epochs

    if args.meta_loss_weight is not None:
        config['meta_loss_weight'] = args.meta_loss_weight
    if args.force_meta_epoch is not None:
        config['force_meta_epoch'] = args.force_meta_epoch
    if args.meta_update_freq is not None:
        config['meta_update_freq'] = args.meta_update_freq
    # neg_sample_ratio 需要检查 DataProcessor 是否支持，若不支持可先忽略或提示

    # [Compatibility]: If wd (weight decay) is passed from command line, override it
    if hasattr(args, 'wd') and args.wd is not None:
        config['wd'] = args.wd


    if args.mode == 'meta' and args.model_name == 'LightGCN':
        if args.dataset == 'amazon':
            # ---------- 基础架构 ----------
            config['embed_dim'] = 256
            config['n_layers'] = 4

            # ---------- 主模型优化 ----------
            config['lr'] = 0.002               # 原 0.0006 → 提高学习率
            config['wd'] = 5e-5                # 原 0.001 → 降低正则化
            config['cl_lambda'] = 0.0          # 关闭对比学习（可选）

            # ---------- 损失函数超参数（在 _compute_losses_meta 中生效）----------
            config['margin_amazon'] = 0.05     # 原 0.15 → 减小 margin
            config['reg_val_amazon'] = 5e-4    # 原 2e-3 → 降低正则强度
            config['cl_temp_amazon'] = 0.5     # 原 0.2 → 提高对比学习温度
            config['cl_eps_amazon'] = 0.05     # 原 0.15 → 减小扰动

            # ---------- 元学习核心参数 ----------
            config['meta_lr'] = 5e-4           # 原 1e-5 → 大幅提高元学习率
            config['meta_update_freq'] = 3     # 原 10 → 更频繁更新元网络
            config['force_meta_epoch'] = 1     # 原 30 → 强制立即激活元学习

            # ---------- 前瞻式元学习参数 ----------
            config['disable_lookahead'] = False
            config['lookahead_switch_epoch'] = None
            config['multi_step_lookahead'] = 3
            config['proactive_hvp_eps'] = 0.005            # 原 1e-4 → 增大虚拟步长
            config['adaptive_epsilon'] = False             # 固定步长，避免自动缩小
            config['proactive_meta_loss_weight'] = 0.02
            config['proactive_loss_explosion_ratio'] = 20.0

            # ---------- 稳定性与泛化 ----------
            config['grad_clip_norm'] = 2.0
            config['use_ema'] = True
            config['ema_decay'] = 0.999
            config['use_warmup'] = True
            config['warmup_epochs'] = 5
            config['batch_size'] = 1024      # 原 2048 → 减小 batch size（显存允许则保留）

            # ---------- 其他必要参数（保持原样或合理默认）----------
            config['meta_max_steps'] = 6
            config['meta_loss_weight'] = 0.01
            config['hvp_eps'] = 1e-4
            config['hvp_max_eps'] = 1e-5
            config['proactive_hvp_max_eps'] = 1e-4
            config['meta_warmup_epochs'] = 0

            if config['epochs'] < 300:
                config['epochs'] = 300

            print(f"   embed_dim: {config['embed_dim']}, n_layers: {config['n_layers']}")
            print(f"   lr: {config['lr']}, meta_lr: {config['meta_lr']}")
            print(f"   disable_lookahead: {config['disable_lookahead']}")
            print(f"   multi_step_lookahead: {config.get('multi_step_lookahead', 1)}")
            print(f"   epochs: {config['epochs']}")
            print(f"   use_ema: {config.get('use_ema', False)}, use_warmup: {config.get('use_warmup', False)}")
        else:
            # ========== 应用自动搜索得到的最佳参数（NDCG@20=0.0579）==========
            # 基础模型参数
            config['embed_dim'] = 256
            config['n_layers'] = 3
            config['lr'] = 0.0037977679442478553
            config['cl_lambda'] = 0.16073441537982291
            config['wd'] = 1e-4
            config['dropout'] = 0.0

            # 元学习核心参数
            config['meta_lr'] = 5.575453980775358e-08
            config['meta_loss_weight'] = 0.044842831927518936
            config['proactive_meta_loss_weight'] = 0.027888427611951233
            config['meta_warmup_epochs'] = 103
            config['multi_step_lookahead'] = 3  # 前瞻步数
            config['meta_update_freq'] = 10
            config['proactive_hvp_eps'] = 2.1387290754148906e-07
            config['force_meta_epoch'] = 107

            # 数据集自适应参数（用于 _compute_losses_meta）
            config['reg_val_yelp'] = 5.3167142741246e-05
            config['cl_temp_yelp'] = 0.42720590636899725
            config['cl_eps_yelp'] = 0.17910958748845152

            # 其他稳定性参数
            config['hvp_eps'] = config['proactive_hvp_eps']
            config['proactive_hvp_max_eps'] = config['proactive_hvp_eps']
            config['hvp_max_eps'] = config['proactive_hvp_eps']
            config['grad_clip_norm'] = 1.0
            config['proactive_loss_explosion_ratio'] = 12.123391106782762  # 保留原搜索值
            config['loss_explosion_threshold'] = 2.0
            config['force_meta_patience'] = 12
            config['meta_decrease_window'] = 4
            config['meta_max_steps'] = 5
            config['fixed_meta_val_batch'] = False
            config['adaptive_epsilon'] = True
            config['disable_lookahead'] = False  # 默认启用前瞻（full_proactive）
            config['lookahead_switch_epoch'] = None

            # ========== 根据 phase 参数微调（可选，不破坏最佳参数主体）==========
            if args.phase == 'reactive':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = None
            elif args.phase == 'proactive':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = 200
                # 保守覆盖（若希望完全遵循最佳参数，可注释掉下面两行）
                config['proactive_hvp_eps'] = 1e-6
                config['force_meta_epoch'] = 200
            elif args.phase == 'dual':
                config['disable_lookahead'] = True
                config['lookahead_switch_epoch'] = 200
                config['proactive_hvp_eps'] = 1e-6
                config['force_meta_epoch'] = 200
                config['dual_aux'] = True
            elif args.phase == 'full_proactive':
                # 此模式完全契合最佳参数，无需额外覆盖
                config['disable_lookahead'] = False
                config['lookahead_switch_epoch'] = None

            # 确保训练轮数充足
            if config['epochs'] < 400:
                config['epochs'] = 400

            print(f"   embed_dim: {config['embed_dim']}, n_layers: {config['n_layers']}")
            print(f"   meta_lr: {config['meta_lr']}, meta_loss_weight: {config['meta_loss_weight']}")
            print(f"   hvp_eps: {config['hvp_eps']}, cl_lambda: {config['cl_lambda']}")
            print(f"   epochs: {config['epochs']}")

    # ========== 新增：多步前瞻参数覆盖（对Amazon和Yelp都生效） ==========
    if args.multi_step is not None:
        config['multi_step_lookahead'] = args.multi_step
        print(f"[Optimization] Multi-step lookahead set to {args.multi_step}")

    # ========== 新增：添加优化参数到config（仅当显式启用时） ==========
    if args.enable_ema:
        config['use_ema'] = True
        config['ema_decay'] = 0.999
        print(f"[Optimization] EMA enabled with decay=0.999")

    if args.grad_accum > 1:
        config['grad_accumulation_steps'] = args.grad_accum
        print(f"[Optimization] Gradient accumulation enabled with {args.grad_accum} steps")

    if args.enable_warmup:
        config['warmup_epochs'] = args.warmup_epochs
        config['use_warmup'] = True
        print(f"[Optimization] Learning rate warmup enabled for {args.warmup_epochs} epochs")

    # ========== 仅针对 gradnorm 模式进行特殊优化 ==========
    if args.mode == 'gradnorm':
        # 专门为 GradNorm 调整的超参数
        config['lr'] = args.lr if args.lr is not None else 0.001
        config['grad_clip_norm'] = 1.0
        config['wd'] = args.wd if args.wd is not None else 1e-4
        config['fast_validation'] = False  # 关闭快速验证，获得真实的 NDCG
        if args.dataset == 'amazon':
            config['grad_clip_norm'] = 1.5
            config['lr'] = 0.0005
        print(f"[GradNorm] Optimized settings: lr={config['lr']}, clip={config['grad_clip_norm']}, wd={config['wd']}, fast_val=OFF")

    # ========== pcgrad 和 mgda 保持原有默认（不做额外修改） ==========
    elif args.mode in ['pcgrad', 'mgda']:
        # 保持原有逻辑：如果命令行未指定，才设置默认值（不强制关闭 fast_validation）
        if args.lr is None:
            config.setdefault('lr', 0.0005)
        else:
            config['lr'] = args.lr
        config.setdefault('grad_clip_norm', 0.5)
        config.setdefault('wd', 1e-4)
        # 注意：不主动修改 config['fast_validation']，保留命令行传入的值（默认 True）
        print(f"[PCGrad/MGDA] Using default settings: lr={config['lr']}, clip={config['grad_clip_norm']}, wd={config['wd']}, fast_val={config.get('fast_validation', True)}")

    # 在加载配置后覆盖
    if args.lr is not None:
        config['lr'] = args.lr

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size


    print(f"[Expert Recommendation] Training {args.model_name} on {args.dataset} in {args.mode} mode.")
    print(f"Configurations: {config}")

    # =================Data Loading =================
    # dp = DataProcessor(args.data_path, dataset_type=args.dataset, batch_size=config['batch_size'])

    # =================Data Loading =================
    data_config = {
        'data_path': args.data_path,
        'dataset': args.dataset,
        'batch_size': config['batch_size'],
    }
    if args.phase == 'dual':
        data_config['dual_aux'] = True
    if args.simulation:
        data_config['simulation_conflict'] = True
    dp = DataProcessor(data_config)

    # =================Initialize Model and Trainer =================
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
        # [Core Change]: Override default learning rate only for SimGCL
        config['lr'] = 0.001   #0.02改为0.001
        config['cl_lambda'] = 0.1
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

    print(f"[INFO] Initializing IMNetTrainer...")
    trainer = IMNetTrainer(dp, model, config, args.device)

    # =================Training Loop =================
    best_recall = 0.0
    best_ndcg = 0.0
    best_epoch = 0
    history = []
    start_epoch = 1
    eval_freq = config.get('eval_freq', 1)  # 【新增】从配置中读取评估频率 / [NEW] Read evaluation frequency from config

    if args.resume:
        checkpoint_path = f"best_{args.dataset}_{args.model_name}_{args.mode}.pth"
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))

    print("[INFO] Start Training...")
    max_epochs = config['epochs']
    for epoch in range(start_epoch, max_epochs + 1):
        start_time = time.time()
        # Train one Epoch
        loss = trainer.train_epoch(epoch=epoch, mode=args.mode)

        # 【核心修改】适配新的评估和动态激活流程
        # [CORE CHANGE] Adapt to the new evaluation and dynamic activation flow
        if epoch % eval_freq == 0 or epoch == 1:
            # 如果启用了EMA，可以选择使用EMA模型评估（通常在训练后期效果更好）
            if args.enable_ema and epoch > config.get('warmup_epochs', 0):
                recall, ndcg = trainer.evaluate_with_ema(top_k=20)
            else:
                recall, ndcg = trainer.evaluate(top_k=20)
            if hasattr(trainer, 'update_lr'):
                # 传入 ndcg 作为性能指标
                trainer.update_lr(ndcg, epoch)
        else:
            recall, ndcg = 0.0, 0.0

        if ndcg > best_ndcg:
            best_recall = recall
            best_ndcg = ndcg
            best_epoch = epoch
            torch.save(model.state_dict(), f"best_{args.dataset}_{args.model_name}_{args.mode}.pth")
            # 如果启用了EMA，也保存EMA模型
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

        history.append([epoch, recall, ndcg])

    # =================Results Persistence =================
    res_df = pd.DataFrame(history, columns=['epoch', 'recall', 'ndcg'])
    res_df.to_csv(f"results_{args.dataset}_{args.model_name}_{args.mode}.csv", index=False)
    print(f"[FINAL RESULTS] Best Recall @ 20: {best_recall:.4f}, Best NDCG @ 20: {best_ndcg:.4f} at Epoch {best_epoch}")
    print(
        f"=== [FINAL_RESULT] Dataset: {args.dataset} | Model: {args.model_name} | Mode: {args.mode} | Seed: {args.seed} | Best Recall@20: {best_recall:.4f} | Best NDCG@20: {best_ndcg:.4f} ===")


if __name__ == '__main__':
    main()