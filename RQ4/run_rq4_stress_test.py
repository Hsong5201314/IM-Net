import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy

from data_loaderMeta import DataProcessor
from backbone import NCF
from train_engine import IMNetTrainer
from imnet import IMNet


def get_flat_grads(loss, model, retain_graph=True):
    """计算单个损失的展平梯度，可选保留计算图"""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph, allow_unused=True)
    flat_grads = []
    for g in grads:
        if g is not None:
            flat_grads.append(g.view(-1))
    if not flat_grads:
        return torch.zeros(1, device=loss.device)
    return torch.cat(flat_grads)


@torch.no_grad()
def compute_gradient_alignment(loss1, loss2, model, retain_graph=True):
    g1 = get_flat_grads(loss1, model, retain_graph=retain_graph)
    g2 = get_flat_grads(loss2, model, retain_graph=retain_graph)
    return F.cosine_similarity(g1, g2, dim=0).item()


def run_stress_test(use_full_dataset=True):
    print("=" * 60)
    print("🔬 Starting RQ4: Controlled Stress Test on NCF Backbone")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Warning] Running on CPU – results may be slow. GPU recommended for convergence.")

    # 选择数据集
    if use_full_dataset:
        dataset_name = 'Yelp_LightGCN'   #yelp_processed_for_meta
        max_epochs = 500
        print("[INFO] Using full Yelp dataset (500 epochs)")
    else:
        dataset_name = 'data_mini_yelp'
        max_epochs = 200
        print("[INFO] Using mini Yelp dataset (200 epochs)")

    dp = DataProcessor(dataset_name)

    # ========== 压力测试配置 ==========
    config = {
        'dataset': dataset_name,
        'model_name': 'NCF',
        'lr': 0.001,
        'meta_lr': 0.005,            # 提高元学习率，促进快速适应
        'wd': 1e-4,
        'embed_dim': 64,
        'hvp_eps': 1e-2,             # 虚拟更新步长（稍大以探索）
        'hvp_max_eps': 0.01,
        'meta_loss_weight': 50.0,    # 元损失权重，强化验证损失信号
    }

    # 修改顺序：先运行 meta，再运行 scalarization
    variants = ['meta', 'scalarization']
    variant_names = {'scalarization': 'Reactive Baseline', 'meta': 'IM-Net (Proactive)'}
    results = []

    # ========== 准备元验证集（用于前瞻式元学习） ==========
    # 从训练集中固定抽取一批数据作为元验证集（不参与训练）
    meta_val_batches = []
    val_batch_size = 16  # 取16个batch作为元验证集，平衡稳定性与效率
    for i, batch in enumerate(dp.train_loader):
        if i >= val_batch_size:
            break
        meta_val_batches.append(batch)
    print(f"[Meta] Using {len(meta_val_batches)} batches for meta-validation.")

    for variant in variants:
        print(f">>>>> Running Stress Test for: {variant_names[variant]} <<<<<")

        model = NCF(dp.n_users, dp.n_items, config['embed_dim']).to(device)
        optimizer_model = torch.optim.Adam(model.parameters(), lr=config['lr'])
        imnet = IMNet(num_tasks=2).to(device)
        optimizer_meta = torch.optim.Adam(imnet.parameters(), lr=config['meta_lr'])

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_alignments = []
            epoch_effective_align = []
            epoch_aux_weights = []

            pbar = tqdm(dp.train_loader, desc=f"Epoch {epoch} [{variant_names[variant]}]", leave=False)
            for batch in pbar:
                users, pos_items, neg_items = [b.to(device).view(-1) for b in batch]
                all_users_emb, all_items_emb = model.get_all_embeddings()
                batch_u = all_users_emb[users]
                batch_pos = all_items_emb[pos_items]
                batch_neg = all_items_emb[neg_items]

                pos_scores = torch.sum(batch_u * batch_pos, dim=1)
                neg_scores = torch.sum(batch_u * batch_neg, dim=1)
                main_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                conflict_loss = -F.logsigmoid(neg_scores - pos_scores).mean()
                if variant == 'scalarization':
                    alignment = compute_gradient_alignment(main_loss, conflict_loss, model, retain_graph=True)
                    epoch_alignments.append(alignment)
                    final_loss = main_loss + conflict_loss
                    optimizer_model.zero_grad()
                    final_loss.backward()
                    optimizer_model.step()
                    aux_weight = 1.0
                    effective_alignment = alignment

                elif variant == 'meta':
                    # 1. 元网络生成权重（不 detach）
                    meta_input = torch.stack([main_loss.detach(), conflict_loss.detach()])
                    raw_weights, weights = imnet(meta_input)
                    weights = torch.sigmoid(weights) * 0.99 + 0.005

                    # 2. 加权训练损失
                    model_loss = weights[0] * main_loss + weights[1] * conflict_loss

                    # 3. 有效对齐（监控）
                    model_grad = get_flat_grads(model_loss, model, retain_graph=True)
                    main_grad = get_flat_grads(main_loss, model, retain_graph=True)
                    effective_alignment = F.cosine_similarity(model_grad, main_grad, dim=0).item()
                    epoch_effective_align.append(effective_alignment)

                    # 4. 原始对齐（监控）
                    weighted_main = weights[0] * main_loss
                    weighted_aux = weights[1] * conflict_loss
                    raw_alignment = compute_gradient_alignment(weighted_main, weighted_aux, model, retain_graph=True)
                    epoch_alignments.append(raw_alignment)

                    # ---------- 前瞻式更新（基于元验证集） ----------
                    saved_params = [p.data.clone() for p in model.parameters()]

                    # 虚拟更新步长（动态）
                    train_grads = torch.autograd.grad(model_loss, model.parameters(), retain_graph=True, allow_unused=True)
                    train_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(train_grads, model.parameters())]
                    grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in train_grads]), 2)
                    base_eps = config.get('hvp_eps', 0.02)
                    safe_eps = base_eps / (grad_norm + 1e-8)
                    safe_eps = torch.clamp(safe_eps, min=1e-6, max=config.get('hvp_max_eps', 0.01))
                    with torch.no_grad():
                        for param, g in zip(model.parameters(), train_grads):
                            param.add_(safe_eps * g)

                    # 计算虚拟更新后的验证损失
                    val_losses_after = []
                    for val_batch in meta_val_batches:
                        v_users, v_pos, v_neg = [b.to(device).view(-1) for b in val_batch]
                        v_u_emb, v_i_emb = model.get_all_embeddings()
                        v_pos_scores = torch.sum(v_u_emb[v_users] * v_i_emb[v_pos], dim=1)
                        v_neg_scores = torch.sum(v_u_emb[v_users] * v_i_emb[v_neg], dim=1)
                        v_main = -F.logsigmoid(v_pos_scores - v_neg_scores).mean()
                        val_losses_after.append(v_main)
                    val_main_loss_after = torch.stack(val_losses_after).mean()

                    # 回滚模型参数
                    with torch.no_grad():
                        for param, saved in zip(model.parameters(), saved_params):
                            param.data = saved

                    # 计算虚拟更新前的验证损失（回滚后的模型）
                    val_losses_before = []
                    for val_batch in meta_val_batches:
                        v_users, v_pos, v_neg = [b.to(device).view(-1) for b in val_batch]
                        v_u_emb, v_i_emb = model.get_all_embeddings()
                        v_pos_scores = torch.sum(v_u_emb[v_users] * v_i_emb[v_pos], dim=1)
                        v_neg_scores = torch.sum(v_u_emb[v_users] * v_i_emb[v_neg], dim=1)
                        v_main = -F.logsigmoid(v_pos_scores - v_neg_scores).mean()
                        val_losses_before.append(v_main)
                    val_main_loss_before = torch.stack(val_losses_before).mean()

                    # 元目标：验证损失的相对下降率（希望为正）
                    rel_improvement = (val_main_loss_before - val_main_loss_after) / (val_main_loss_before + 1e-8)
                    # 元损失：负的下降率（越小表示下降越多）
                    meta_loss = -rel_improvement

                    # 动态权重惩罚（随 epoch 线性增长）
                    dynamic_lambda = config.get('meta_loss_weight', 50.0) * (1 + epoch / 100)
                    weight_penalty = weights[1] + weights[1]**2
                    # 最终损失 = 训练损失 + 惩罚项 + 元损失
                    final_loss = model_loss + dynamic_lambda * (meta_loss + weight_penalty)

                    # 反向传播
                    optimizer_model.zero_grad()
                    optimizer_meta.zero_grad()
                    final_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(imnet.parameters(), max_norm=1.0)

                    optimizer_model.step()
                    optimizer_meta.step()

                    aux_weight = weights[1].item()
                    epoch_aux_weights.append(aux_weight)
                    pbar.set_postfix({'ρ_eff': f"{effective_alignment:.2f}", 'W_harm': f"{aux_weight:.2f}"})

            # ========== 评估 ==========
            trainer_shell = IMNetTrainer(dp, model, config, device)
            recall, ndcg = trainer_shell.evaluate(top_k=20)

            if variant == 'scalarization':
                avg_alignment = np.mean(epoch_alignments) if epoch_alignments else 0.0
                avg_effective = avg_alignment
                avg_aux_weight = 1.0
            else:
                avg_alignment = np.mean(epoch_alignments) if epoch_alignments else 0.0
                avg_effective = np.mean(epoch_effective_align) if epoch_effective_align else 0.0
                avg_aux_weight = np.mean(epoch_aux_weights) if epoch_aux_weights else 0.0

            print(f"Epoch {epoch:02d} | Recall: {recall:.4f} | Raw ρ: {avg_alignment:.3f} | Eff ρ: {avg_effective:.3f} | AuxW: {avg_aux_weight:.3f}")

            results.append({
                'Method': variant_names[variant],
                'Epoch': epoch,
                'Recall': recall,
                'GradientAlignment': avg_alignment,
                'EffectiveAlignment': avg_effective,
                'AuxWeight': avg_aux_weight
            })

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/rq4_stress_test_data.csv', index=False)
    plot_stress_test_figures(df)


def plot_stress_test_figures(df):
    """生成符合顶级期刊风格的三联图"""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    colors = {'Reactive Baseline': '#B0BEC5', 'IM-Net (Proactive)': '#E53935'}

    # (a) 性能轨迹
    sns.lineplot(data=df, x='Epoch', y='Recall', hue='Method', palette=colors, lw=2.5, ax=axes[0])
    axes[0].set_title('a. Performance Under Stress (NCF)', fontweight='bold')
    axes[0].set_ylabel('Validation Recall@20')
    axes[0].set_ylim(bottom=0)

    # (b) 有效梯度对齐轨迹
    if 'EffectiveAlignment' in df.columns:
        align_col = 'EffectiveAlignment'
        ylabel = 'Cosine Similarity ρ(g_model, g_main)'
    else:
        align_col = 'GradientAlignment'
        ylabel = 'Cosine Similarity ρ(g_main, g_conflict)'
    sns.lineplot(data=df, x='Epoch', y=align_col, hue='Method', palette=colors, lw=2.5, ax=axes[1])
    axes[1].set_title('b. Effective Gradient Alignment', fontweight='bold')
    axes[1].set_ylabel(ylabel)
    axes[1].axhline(0.0, color='grey', linestyle='--', alpha=0.5)
    axes[1].axhline(-1.0, color='lightgrey', linestyle=':', alpha=0.5)

    # (c) 有害任务权重轨迹（仅 IM-Net）
    df_imnet = df[df['Method'] == 'IM-Net (Proactive)']
    if not df_imnet.empty:
        axes[2].plot(df_imnet['Epoch'], df_imnet['AuxWeight'], color=colors['IM-Net (Proactive)'], lw=2.5)
        axes[2].set_title('c. IM-Net: Harmful Task Weight', fontweight='bold')
        axes[2].set_ylabel('Weight of Conflicting Task')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylim(0, 1)
        axes[2].axhline(0.01, color='red', linestyle='--', alpha=0.7, label='Lower bound (0.01)')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig('results/Fig_RQ4_Stress_Test.png', dpi=300)
    plt.savefig('results/Fig_RQ4_Stress_Test.pdf', bbox_inches='tight')
    print("[INFO] Stress test figures saved to results/Fig_RQ4_Stress_Test.png")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Use full Yelp dataset')
    args = parser.parse_args()
    run_stress_test(use_full_dataset=args.full)