import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import copy
from utils import compute_gradient_conflict_metrics, compute_hessian_spectral_radius, HessianTracker


class IMNetTrainer:

    def __init__(self, data_processor, model, config, device):
        self.dp = data_processor
        self.device = device
        self.config = config
        self.model = model

        self.model_name = config.get('model_name', 'LightGCN')
        self.is_yelp    = (self.config.get('dataset') == 'yelp')
        self.is_amazon  = (self.config.get('dataset') == 'amazon')
        self.is_lightgcn_meta = (self.model_name == 'LightGCN')
        self.mode             = config.get('mode', 'meta')
        self.is_random_dynamic = (self.mode == 'random_dynamic')

        # ========== 仿真模式标志 ==========
        self.simulation_mode  = config.get('simulation_mode', False)
        self.imnet_input_dim  = config.get('imnet_input_dim', 4)
        self.hessian_freq     = config.get('hessian_freq', 10)
        self.rho_target       = config.get('rho_target', -0.99)
        self.grad_force_enabled = self.simulation_mode

        self.lambda_quad = config.get('lambda_quad', 1.0)

        # ========== 基于虚拟更新的元学习配置 ==========
        self.lambda_conflict       = config.get('lambda_conflict', 0.0)
        self.lambda_sparsity       = config.get('lambda_sparsity', 0.0)
        self.lambda_interference   = config.get('lambda_interference', 100.0)
        self.interference_threshold = config.get('interference_threshold', 0.0)
        self.inner_lr = config.get('inner_lr', 0.02)  # 降低虚拟步长默认值
        self.conflict_ema          = None

        # ================= 动态元学习控制 =================
        self.meta_active            = False
        self.meta_cool_down         = 0
        self.consecutive_decreases  = 0
        self.best_metric            = 0.0
        self.best_model_state       = None
        self.pre_intervention_state = None
        self.meta_intervention_steps = 0
        self.meta_max_steps         = config.get('meta_max_steps', 5)
        self.meta_recovery_threshold = config.get('meta_recovery_threshold', 0.001)
        self.meta_decrease_window   = config.get('meta_decrease_window', 3)

        # ================= 禁用前瞻 =================
        self.disable_lookahead          = config.get('disable_lookahead', False)
        self.disable_lookahead_original = self.disable_lookahead
        self.current_disable_lookahead  = self.disable_lookahead
        self.lookahead_switch_epoch     = config.get('lookahead_switch_epoch', None)
        if self.lookahead_switch_epoch is not None:
            print(
                f"[Dynamic Mode] Reactive before epoch {self.lookahead_switch_epoch}, "
                f"then switch to Proactive."
            )

        if self.simulation_mode:
            config['meta_lr']  = 5e-4
            config['wd']       = 5e-4
            config['dropout']  = 0.1

        # ================= 损失爆炸检测 =================
        self.avg_loss_before_meta = None

        # ================= 保底触发参数 =================
        self.last_best_update_epoch = 0
        self.force_meta_patience    = config.get('force_meta_patience', 15)
        self.force_meta_start_epoch = config.get('force_meta_start_epoch', 50)
        self.force_meta_epoch       = config.get('force_meta_epoch', None)

        # 早停参数
        self.early_stop_patience = config.get('early_stop_patience', 0)
        self.no_improve_epochs   = 0

        # ================= HVP 步长控制 =================
        self.hvp_eps     = config.get('hvp_eps', 0.01)
        self.hvp_max_eps = config.get('hvp_max_eps', 5e-4)

        # ================= 前瞻模式专用稳定参数 =================
        self.proactive_hvp_eps          = config.get('proactive_hvp_eps', self.hvp_eps)
        self.proactive_hvp_eps_original = self.proactive_hvp_eps
        self.proactive_hvp_max_eps      = config.get('proactive_hvp_max_eps', self.hvp_max_eps)
        self.proactive_meta_loss_weight = config.get(
            'proactive_meta_loss_weight', config.get('meta_loss_weight', 0.1)
        )
        self.adaptive_epsilon     = config.get('adaptive_epsilon', True)
        self.epsilon_base         = config.get('epsilon_base', 1e-5)
        self.multi_step_lookahead = config.get('multi_step_lookahead', 1)
        self.proactive_loss_explosion_ratio = config.get('proactive_loss_explosion_ratio', 10.0)

        # ================= EMA 支持 =================
        self.use_ema = config.get('use_ema', True)
        if self.use_ema:
            self.ema_decay = config.get('ema_decay', 0.995)
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            print(f"[EMA] EMA enabled with decay={self.ema_decay}")

        # ================= 梯度累积支持 =================
        self.grad_accumulation_steps = config.get('grad_accumulation_steps', 1)
        if self.grad_accumulation_steps > 1:
            print(
                f"[GradAccum] Gradient accumulation enabled with "
                f"{self.grad_accumulation_steps} steps"
            )

        # ================= 学习率预热 =================
        self.use_warmup    = config.get('use_warmup', False)
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.initial_lr    = config.get('lr', 0.001)
        if self.use_warmup and self.warmup_epochs > 0:
            print(f"[Warmup] Learning rate warmup enabled for {self.warmup_epochs} epochs")
            self.current_warmup_epoch = 0

        # ================= 初始化 IMNet =================
        from imnet import IMNet

        if self.simulation_mode:
            # ── 仿真模式：输入维度 5（4个损失 + 干涉能）
            self.imnet = IMNet(
                num_tasks=4, hidden_dim=128, input_dim=self.imnet_input_dim
            ).to(device)
            self.hessian_tracker = HessianTracker(
                model=self.model,
                compute_freq=self.hessian_freq,
                num_iter=20,
                loss_scaling=1.0
            )
            self.lambda_spectral = config.get('lambda_spectral', 0.05)
            self.simulation_metrics = {
                'epoch':              [],
                'batch':              [],
                'interference_energy':[],
                'grad_cos_sim':       [],
                'w_aux':              [],
                'w_main':             [],
                'spectral_radius':    []
            }
            print(f"[SIMULATION] IMNet input dimension set to {self.imnet_input_dim}")
        else:
            self.imnet = IMNet(num_tasks=4).to(device)
            self.lambda_spectral = config.get('lambda_spectral', 0.0)
            if self.lambda_spectral > 0:
                self.hessian_tracker = HessianTracker(
                    model=self.model,
                    compute_freq=config.get('hessian_freq', 10),
                    num_iter=20,
                    loss_scaling=1.0
                )
                print(
                    f"[Init] HessianTracker initialized with "
                    f"lambda_spectral={self.lambda_spectral}"
                )
            else:
                self.hessian_tracker = None

        # ================= 必要配置项 =================
        if 'grad_clip_norm' not in self.config:
            self.config['grad_clip_norm'] = 1.0
            print(f"[Init] Set default grad_clip_norm to 1.0")
        if 'loss_explosion_threshold' not in self.config:
            self.config['loss_explosion_threshold'] = 100.0
            print(f"[Init] Set default loss_explosion_threshold to 3.0")

        # 固定元验证 batch（可选）
        if self.config.get('fixed_meta_val_batch', False):
            try:
                self.fixed_meta_val_batch = next(iter(self.dp.meta_val_loader))
                print("[Expert Info] Using fixed meta-validation batch.")
            except StopIteration:
                print(
                    "[WARNING] Could not create fixed meta-val batch, "
                    "fallback to random sampling."
                )
                self.fixed_meta_val_batch = None
        else:
            self.fixed_meta_val_batch = None

        # ================= 优化器 =================
        wd      = config.get('wd', 5e-4)
        dropout = config.get('dropout', 0.1)
        if self.model_name == 'NCF':
            self.optimizer_model = optim.AdamW(
                self.model.parameters(), lr=config['lr'], weight_decay=wd
            )
        else:
            self.optimizer_model = optim.AdamW(
                self.model.parameters(), lr=config['lr']
            )
        self.optimizer_meta = optim.AdamW(
            self.imnet.parameters(), lr=config.get('meta_lr', 5e-4)
        )

        # ================= 学习率调度器 =================
        if not self.disable_lookahead:
            self.scheduler_model = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_model, T_0=50, T_mult=2, eta_min=1e-5
            )
            self.scheduler_meta = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_meta, T_0=50, T_mult=2, eta_min=1e-7
            )
        else:
            if self.is_lightgcn_meta and self.is_yelp:
                self.scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_model, mode='max', factor=0.5, patience=12
                )
                self.scheduler_meta = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_meta, mode='max', factor=0.5, patience=8
                )
            else:
                self.scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_model, mode='max', factor=0.5, patience=8
                )
                self.scheduler_meta = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_meta, mode='max', factor=0.5, patience=5
                )

        # ================= 元验证集加载器 =================
        if hasattr(self.dp, 'meta_val_loader') and self.dp.meta_val_loader is not None:
            self.meta_val_loader = iter(self.dp.meta_val_loader)
            print("[Expert Info] Using provided meta_val_loader.")
        else:
            print(
                "[WARNING] meta_val_loader is None. "
                "Attempting to create a proxy Meta-Val set from train_loader."
            )
            all_train_batches = list(self.dp.train_loader)
            val_size = max(1, len(all_train_batches) // 10)
            self.meta_val_samples = all_train_batches[-val_size:]
            self.meta_val_loader  = iter(self.meta_val_samples)
            print(
                f"[Critical Fix] Created a proxy Meta-Val set of "
                f"{len(self.meta_val_samples)} batches."
            )

        # 缓存训练批次列表
        self._train_batches_cache = None

        # 低频更新控制
        self.meta_update_freq  = config.get('meta_update_freq', 5)
        self.meta_batch_counter = 0

        # ================= 验证损失缓存 =================
        self.cached_val_loss  = None
        self.cached_val_step  = -100
        self.val_cache_freq   = config.get('val_cache_freq', 10)

        self.use_fast_validation = config.get('fast_validation', True)
        if self.use_fast_validation:
            self.val_batch_count      = 1
            self.cached_val_loss      = None
            self.cached_val_step      = -100
            self.val_cache_freq       = config.get('val_cache_freq', 10)
            self.cached_first_val_loss = None
            self.cached_last_val_loss  = None
            print(
                f"[Optimization] Fast validation: {self.val_batch_count} batch(es), "
                f"cache every {self.val_cache_freq} steps"
            )
        else:
            self.val_batch_count       = 5
            self.cached_val_loss       = None
            self.val_cache_freq        = 1
            self.cached_first_val_loss = None
            self.cached_last_val_loss  = None

        # ================= _step_gradnorm 属性初始化 =================
        self.initial_losses     = None
        self.gradnorm_weights   = None
        self.gradnorm_optimizer = None

        # ================= GradReverse 算子 =================
        class GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambda_):
                ctx.lambda_ = lambda_
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.lambda_ * grad_output, None

        self.GradReverse          = GradReverse
        self.grad_reverse_lambda  = 0.0
        self.sim_patience_counter = 0
        self.current_epoch        = 0

        # 额外的 EMA 缓存
        self.model_loss_ema = None
        self.meta_loss_ema  = None
        self.fixed_val_batches = []

        # 确保 fixed_val_batches 非空
        if hasattr(self.dp, 'meta_val_loader') and self.dp.meta_val_loader is not None:
            try:
                self.fixed_val_batches = [next(iter(self.dp.meta_val_loader))]
            except StopIteration:
                pass
        if not self.fixed_val_batches:
            try:
                self.fixed_val_batches = [list(self.dp.train_loader)[-1]]
            except Exception:
                pass



    # ================= EMA更新方法 =================
    def update_ema(self, decay=None):
        if not self.use_ema:
            return
        if decay is None:
            decay = self.ema_decay
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    # ================= 评估方法 =================
    @torch.no_grad()
    def evaluate_with_ema(self, top_k=20):
        if not self.use_ema:
            return self.evaluate(top_k)
        self.ema_model.eval()
        from main_gpu import get_metrics
        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp, 'norm_adj') else None
        recall, ndcg = get_metrics(
            self.ema_model, self.config['model_name'], adj,
            self.dp.train_dict, self.dp.test_dict, self.device, top_k=top_k
        )
        return recall, ndcg

    @torch.no_grad()
    def evaluate(self, top_k=20):
        self.model.eval()
        from main_gpu import get_metrics
        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp, 'norm_adj') else None
        recall, ndcg = get_metrics(
            self.model, self.config['model_name'], adj,
            self.dp.train_dict, self.dp.test_dict, self.device, top_k=top_k
        )
        return recall, ndcg

    # ================= 学习率预热 =================
    def apply_warmup(self, epoch):
        if not self.use_warmup or self.warmup_epochs == 0:
            return
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            current_lr = self.initial_lr * warmup_factor
            for param_group in self.optimizer_model.param_groups:
                param_group['lr'] = current_lr
            if epoch != self.current_warmup_epoch:
                self.current_warmup_epoch = epoch
                print(f"[Warmup] Epoch {epoch}: LR = {current_lr:.6f}")

    # ================= 梯度冲突强制器 =================
    def _force_gradients(self, g_primary, g_aux):
        flat_pri = torch.cat([g.view(-1) for g in g_primary])
        flat_aux = torch.cat([g.view(-1) for g in g_aux])
        norm_pri = flat_pri.norm() + 1e-8
        norm_aux = flat_aux.norm() + 1e-8
        proj = torch.dot(flat_pri, flat_aux) / (norm_pri * norm_aux + 1e-8)
        orth = flat_aux - proj * flat_pri
        orth_norm = orth.norm() + 1e-8
        target_cos = self.rho_target
        a = target_cos / norm_pri
        b = (1 - target_cos**2)**0.5 / orth_norm
        new_flat = a * flat_pri + b * orth
        scale = norm_aux / (new_flat.norm() + 1e-8)
        new_flat = new_flat * scale
        new_g_aux = []
        idx = 0
        for g in g_aux:
            numel = g.numel()
            new_g_aux.append(new_flat[idx:idx+numel].view(g.shape))
            idx += numel
        return new_g_aux

    # ================= 损失计算 =================
    def _compute_losses_and_grads(self, batch):
        if self.simulation_mode:
            users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]
            is_amazon = self.is_amazon
            is_yelp = self.is_yelp
            model_name = self.model_name
            adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp,
                                                              'norm_adj') and self.dp.norm_adj is not None else None
            if self.is_lightgcn_meta and is_yelp:
                margin = self.config.get('margin_yelp', 0.05)
                reg_val = self.config.get('reg_val_yelp', 5e-4)
            elif self.is_lightgcn_meta and is_amazon:
                margin = self.config.get('margin_amazon', 0.15)
                reg_val = self.config.get('reg_val_amazon', 2e-3)
            else:
                margin = 0.0
                reg_val = self.config.get('wd', 5e-4)

            if model_name == 'SimGCL':
                u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            elif model_name == 'NCF':
                u_emb, i_emb = self.model.get_all_embeddings()
            else:
                u_emb, i_emb = self.model.get_all_embeddings(adj)

            if self.meta_active and self.model.training:
                if self.is_amazon:
                    u_emb = F.dropout(u_emb, p=0.4, training=True)
                    i_emb = F.dropout(i_emb, p=0.4, training=True)
                else:
                    u_emb = F.dropout(u_emb, p=0.2, training=True)
                    i_emb = F.dropout(i_emb, p=0.2, training=True)

            lambda_rev = getattr(self, 'grad_reverse_lambda', 0.0)
            u_emb_rev = self.GradReverse.apply(u_emb, lambda_rev)
            i_emb_rev = self.GradReverse.apply(i_emb, lambda_rev)

            batch_u_main = u_emb[users]
            batch_pos_main = i_emb[pos_items]
            batch_neg_main = i_emb[neg_items]
            pos_scores_main = torch.sum(batch_u_main * batch_pos_main, dim=1)
            neg_scores_main = torch.sum(batch_u_main * batch_neg_main, dim=1)
            if self.is_lightgcn_meta:
                bpr_loss = -F.logsigmoid(pos_scores_main - neg_scores_main - margin).mean()
            else:
                bpr_loss = -F.logsigmoid(pos_scores_main - neg_scores_main).mean()
            reg_loss = (1 / 2) * (
                        batch_u_main.norm(2).pow(2) + batch_pos_main.norm(2).pow(2) + batch_neg_main.norm(2).pow(
                    2)) / float(users.shape[0])
            main_loss = bpr_loss + reg_val * reg_loss

            conflict_scale = self.config.get('conflict_scale', 1.0)
            batch_u_aux = u_emb_rev[users]
            batch_pos_aux = i_emb_rev[pos_items]
            batch_neg_aux = i_emb_rev[neg_items]
            pos_scores_aux = torch.sum(batch_u_aux * batch_pos_aux, dim=1)
            neg_scores_aux = torch.sum(batch_u_aux * batch_neg_aux, dim=1)
            aux_loss = F.logsigmoid(neg_scores_aux - pos_scores_aux).mean() * conflict_scale

            aux_loss2 = torch.tensor(0.0, device=self.device)
            cl_loss   = torch.tensor(0.0, device=self.device)
            losses    = torch.stack([main_loss, aux_loss, aux_loss2, cl_loss])
            return losses, None, None
        else:
            losses, _ = self._compute_losses_meta(batch)
            return losses, None, None

    def _nash_mtl_gradient(self, task_grads_flat, max_iter=20, lr=0.1):
        num_tasks = len(task_grads_flat)
        G = torch.zeros(num_tasks, num_tasks, device=task_grads_flat[0].device)
        for i in range(num_tasks):
            for j in range(num_tasks):
                G[i, j] = torch.dot(task_grads_flat[i], task_grads_flat[j])
        λ = torch.ones(num_tasks, device=G.device) / num_tasks
        for _ in range(max_iter):
            grad = torch.mv(G, λ)
            λ_new = λ - lr * grad
            λ_new = torch.clamp(λ_new, min=0)
            λ_new = λ_new / (λ_new.sum() + 1e-8)
            λ = λ_new
        combined = sum(λ[i] * task_grads_flat[i] for i in range(num_tasks))
        return combined, λ

    def _compute_losses_meta(self, batch):
        users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]
        is_amazon = self.is_amazon
        is_yelp = self.is_yelp
        model_name = self.model_name
        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp,
                                                          'norm_adj') and self.dp.norm_adj is not None else None

        if self.is_lightgcn_meta and is_yelp:
            cl_temp = self.config.get('cl_temp_yelp', 0.3)
            margin = self.config.get('margin_yelp', 0.05)
            reg_val = self.config.get('reg_val_yelp', 5e-4)
            cl_eps = self.config.get('cl_eps_yelp', 0.05)
        elif self.is_lightgcn_meta and is_amazon:
            cl_temp = self.config.get('cl_temp_amazon', 0.07)
            margin = self.config.get('margin_amazon', 0.15)
            reg_val = self.config.get('reg_val_amazon', 2e-3)
            cl_eps = self.config.get('cl_eps_amazon', 0.15)
        else:
            cl_temp = self.config.get('cl_temp', 0.2)
            margin = 0.0
            reg_val = self.config.get('wd', 1e-4)
            cl_eps = self.config.get('cl_eps', 0.1)

        if model_name == 'SimGCL':
            u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            u_v1, i_v1 = self.model.get_all_embeddings(adj, perturbed=True)
            u_v2, i_v2 = self.model.get_all_embeddings(adj, perturbed=True)
        else:
            u_emb, i_emb = self.model.get_all_embeddings() if model_name == 'NCF' else self.model.get_all_embeddings(
                adj)
            eps = cl_eps
            u_v1, i_v1 = u_emb + torch.randn_like(u_emb) * eps, i_emb + torch.randn_like(i_emb) * eps
            u_v2, i_v2 = u_emb + torch.randn_like(u_emb) * eps, i_emb + torch.randn_like(i_emb) * eps

        if self.meta_active and self.model.training:
            if self.is_amazon:
                u_emb = F.dropout(u_emb, p=0.4, training=True)
                i_emb = F.dropout(i_emb, p=0.4, training=True)
                noise_std = 0.05
                u_emb = u_emb + torch.randn_like(u_emb) * noise_std
                i_emb = i_emb + torch.randn_like(i_emb) * noise_std
            else:
                u_emb = F.dropout(u_emb, p=0.2, training=True)
                i_emb = F.dropout(i_emb, p=0.2, training=True)

        unique_users, unique_items = torch.unique(users), torch.unique(pos_items)
        u_cl_loss = self.info_nce(u_v1[unique_users], u_v2[unique_users], cl_temp)
        i_cl_loss = self.info_nce(i_v1[unique_items], i_v2[unique_items], cl_temp)
        cl_loss = u_cl_loss + i_cl_loss

        batch_u, batch_pos, batch_neg = u_emb[users], i_emb[pos_items], i_emb[neg_items]
        pos_scores = torch.sum(batch_u * batch_pos, dim=1)
        neg_scores = torch.sum(batch_u * batch_neg, dim=1)
        if self.is_lightgcn_meta:
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores - margin).mean()
        else:
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg_loss = (1 / 2) * (batch_u.norm(2).pow(2) + batch_pos.norm(2).pow(2) + batch_neg.norm(2).pow(2)) / float(
            users.shape[0])
        main_loss = bpr_loss + reg_val * reg_loss

        if hasattr(self.dp, 'simulation_conflict') and self.dp.simulation_conflict:
            conflict_scale = self.config.get('conflict_scale', 1.0)
            aux_loss = -F.logsigmoid(pos_scores).mean() * conflict_scale
            aux_loss2 = torch.tensor(0.0, device=self.device)
            cl_loss = torch.tensor(0.0, device=self.device)
            avg_score_diff = (pos_scores - neg_scores).mean()
            return torch.stack([main_loss, aux_loss, aux_loss2, cl_loss]), avg_score_diff

        aux_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.dp, 'aux_links') and self.dp.aux_links is not None and len(self.dp.aux_links) > 0:
            batch_size_aux = users.shape[0]
            idx = np.random.choice(len(self.dp.aux_links), batch_size_aux, replace=True)
            node1 = torch.tensor(self.dp.aux_links[idx, 0], dtype=torch.long, device=self.device)
            node2 = torch.tensor(self.dp.aux_links[idx, 1], dtype=torch.long, device=self.device)
            if is_amazon:
                node1 = node1 % i_emb.size(0)
                node2 = node2 % i_emb.size(0)
                emb1, emb2 = i_emb[node1], i_emb[node2]
            else:
                node1 = node1 % u_emb.size(0)
                node2 = node2 % u_emb.size(0)
                emb1, emb2 = u_emb[node1], u_emb[node2]
            aux_loss = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()

        aux_loss2 = torch.tensor(0.0, device=self.device)
        if self.is_yelp and hasattr(self.dp, 'item_item_links') and len(self.dp.item_item_links) > 0:
            node1, node2 = self.dp.sample_item_aux_links(users.shape[0])
            node1, node2 = node1.to(self.device), node2.to(self.device)
            emb1, emb2 = i_emb[node1], i_emb[node2]
            aux_loss2 = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()
        elif self.is_amazon and hasattr(self.dp, 'user_user_links') and len(self.dp.user_user_links) > 0:
            node1, node2 = self.dp.sample_user_aux_links(users.shape[0])
            node1, node2 = node1.to(self.device), node2.to(self.device)
            emb1, emb2 = u_emb[node1], u_emb[node2]
            aux_loss2 = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()

        avg_score_diff = (pos_scores - neg_scores).mean()
        return torch.stack([main_loss, aux_loss, aux_loss2, cl_loss]), avg_score_diff

    def _compute_losses(self, batch):
        if self.simulation_mode:
            users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]
            is_amazon = self.is_amazon
            is_yelp = self.is_yelp
            model_name = self.model_name
            adj = (self.dp.norm_adj.to(self.device)
                   if hasattr(self.dp, 'norm_adj') and self.dp.norm_adj is not None else None)
            if self.is_lightgcn_meta and is_yelp:
                margin  = self.config.get('margin_yelp', 0.05)
                reg_val = self.config.get('reg_val_yelp', 5e-4)
            elif self.is_lightgcn_meta and is_amazon:
                margin  = self.config.get('margin_amazon', 0.15)
                reg_val = self.config.get('reg_val_amazon', 2e-3)
            else:
                margin  = 0.0
                reg_val = self.config.get('wd', 5e-4)

            if model_name == 'SimGCL':
                u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            elif model_name == 'NCF':
                u_emb, i_emb = self.model.get_all_embeddings()
            else:
                u_emb, i_emb = self.model.get_all_embeddings(adj)

            batch_u_main   = u_emb[users]
            batch_pos_main = i_emb[pos_items]
            batch_neg_main = i_emb[neg_items]
            pos_scores_main = torch.sum(batch_u_main * batch_pos_main, dim=1)
            neg_scores_main = torch.sum(batch_u_main * batch_neg_main, dim=1)
            if self.is_lightgcn_meta:
                bpr_loss = -F.logsigmoid(pos_scores_main - neg_scores_main - margin).mean()
            else:
                bpr_loss = -F.logsigmoid(pos_scores_main - neg_scores_main).mean()
            reg_loss = (
                (1 / 2) * (
                    batch_u_main.norm(2).pow(2)
                    + batch_pos_main.norm(2).pow(2)
                    + batch_neg_main.norm(2).pow(2)
                ) / float(users.shape[0])
            )
            main_loss = bpr_loss + reg_val * reg_loss

            lambda_rev  = getattr(self, 'grad_reverse_lambda', 0.0)
            u_emb_rev   = self.GradReverse.apply(u_emb, lambda_rev)
            i_emb_rev   = self.GradReverse.apply(i_emb, lambda_rev)
            batch_u_aux   = u_emb_rev[users]
            batch_pos_aux = i_emb_rev[pos_items]
            batch_neg_aux = i_emb_rev[neg_items]
            pos_scores_aux = torch.sum(batch_u_aux * batch_pos_aux, dim=1)
            neg_scores_aux = torch.sum(batch_u_aux * batch_neg_aux, dim=1)
            aux_loss = (
                F.logsigmoid(neg_scores_aux - pos_scores_aux).mean()
                * self.config.get('conflict_scale', 1.0)
            )
            aux_loss2 = torch.tensor(0.0, device=self.device)
            cl_loss   = torch.tensor(0.0, device=self.device)
            return torch.stack([main_loss, aux_loss, aux_loss2, cl_loss])

        if hasattr(self.dp, 'simulation_conflict') and self.dp.simulation_conflict:
            users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]
            is_amazon  = (self.config.get('dataset') == 'amazon')
            model_name = self.config.get('model_name', 'LightGCN')
            adj = (self.dp.norm_adj.to(self.device)
                   if hasattr(self.dp, 'norm_adj') and self.dp.norm_adj is not None else None)
            if model_name == 'SimGCL':
                u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            elif model_name == 'NCF':
                u_emb, i_emb = self.model.get_all_embeddings()
            else:
                u_emb, i_emb = self.model.get_all_embeddings(adj)
            batch_u   = u_emb[users]
            batch_pos = i_emb[pos_items]
            batch_neg = i_emb[neg_items]
            pos_scores = torch.sum(batch_u * batch_pos, dim=1)
            neg_scores = torch.sum(batch_u * batch_neg, dim=1)
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = (
                (1 / 2) * (
                    batch_u.norm(2).pow(2)
                    + batch_pos.norm(2).pow(2)
                    + batch_neg.norm(2).pow(2)
                ) / float(users.shape[0])
            )
            main_loss      = bpr_loss + self.config.get('wd', 1e-4) * reg_loss
            conflict_scale = self.config.get('conflict_scale', 1.0)
            aux_loss       = -F.logsigmoid(pos_scores).mean() * conflict_scale
            cl_loss        = torch.tensor(0.0, device=self.device)
            return torch.stack([main_loss, aux_loss, cl_loss])
        else:
            users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]
            is_amazon  = (self.config.get('dataset') == 'amazon')
            model_name = self.config.get('model_name', 'LightGCN')
            adj = (self.dp.norm_adj.to(self.device)
                   if hasattr(self.dp, 'norm_adj') and self.dp.norm_adj is not None else None)
            if model_name == 'SimGCL':
                u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            elif model_name == 'NCF':
                u_emb, i_emb = self.model.get_all_embeddings()
            else:
                u_emb, i_emb = self.model.get_all_embeddings(adj)
            cl_loss = torch.tensor(0.0, device=self.device)
            if model_name == 'SimGCL':
                u_v1, i_v1 = self.model.get_all_embeddings(adj, perturbed=True)
                u_v2, i_v2 = self.model.get_all_embeddings(adj, perturbed=True)
                cl_temp      = self.config.get('cl_temp', 0.2)
                unique_users = torch.unique(users)
                unique_items = torch.unique(pos_items)
                u_cl_loss    = self.info_nce(u_v1[unique_users], u_v2[unique_users], cl_temp)
                i_cl_loss    = self.info_nce(i_v1[unique_items], i_v2[unique_items], cl_temp)
                cl_loss      = u_cl_loss + i_cl_loss
            batch_u, batch_pos, batch_neg = u_emb[users], i_emb[pos_items], i_emb[neg_items]
            pos_scores = torch.sum(batch_u * batch_pos, dim=1)
            neg_scores = torch.sum(batch_u * batch_neg, dim=1)
            bpr_loss   = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss   = (
                (1 / 2) * (
                    batch_u.norm(2).pow(2)
                    + batch_pos.norm(2).pow(2)
                    + batch_neg.norm(2).pow(2)
                ) / float(users.shape[0])
            )
            main_loss = bpr_loss + self.config.get('wd', 1e-4) * reg_loss
            node1, node2 = self.dp.sample_aux_links(users.shape[0])
            node1, node2 = node1.to(self.device), node2.to(self.device)
            emb1 = i_emb[node1] if is_amazon else u_emb[node1]
            emb2 = i_emb[node2] if is_amazon else u_emb[node2]
            aux_loss = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()
            return torch.stack([main_loss, aux_loss, cl_loss])

    def info_nce(self, z1, z2, temp):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        pos_sim = torch.sum(z1 * z2, dim=-1)
        sim_matrix = torch.matmul(z1, z2.t())
        matrix_exp = torch.exp(sim_matrix / temp)
        pos_exp = torch.exp(pos_sim / temp)
        return -torch.log(pos_exp / (matrix_exp.sum(dim=-1) + 1e-8)).mean()

    def _clip_weights(self, w):
        min_main = self.config.get('min_main_weight', 0.45)
        # 先确保 w 是平铺的一维，如果需要可以处理 batch 维度，但这里假设是一维
        w_clipped = w.clone()
        if w_clipped[0] < min_main:
            # 构造新的权重向量
            w_main_clipped = min_main
            w_rest = w_clipped[1:] * ((1 - min_main) / (w_clipped[1:].sum() + 1e-8))
            new_weights = torch.cat([torch.tensor([w_main_clipped], device=w.device), w_rest])
            return new_weights
        else:
            return w_clipped / w_clipped.sum()

    def _compute_meta_val_loss(self):
        """返回当前元验证损失（仅用于诊断）"""
        if not self.fixed_val_batches:
            return 0.0
        val_batch = self.fixed_val_batches[0]
        losses, _, _ = self._compute_losses_and_grads(val_batch)
        # 返回主任务损失作为验证损失的代表
        return losses[0].item()

    def train_epoch(self, epoch, mode='meta'):
        self.current_epoch = epoch
        self.apply_warmup(epoch)
        self.model.train()

        # ========= 诊断：仿真模式下每5个epoch打印一次元验证损失 =========
        if self.config.get('simulation_mode', False) and epoch % 5 == 0:
            try:
                val_loss = self._compute_meta_val_loss()  # 请确保该方法已实现
                print(f"[DEBUG] Meta validation loss at epoch {epoch}: {val_loss:.6f}")
            except Exception as e:
                print(f"[DEBUG] Could not compute meta val loss: {e}")

        # ========= 动态更新固定验证 batch（每5个epoch刷新一次） =========
        if self.simulation_mode and epoch % 5 == 0 and epoch > 0:
            try:
                new_val_batch = next(self.meta_val_loader)
                self.fixed_val_batches = [new_val_batch]
                print(f"[Meta] Refreshed validation batch at epoch {epoch}")
            except StopIteration:
                self.meta_val_loader = iter(self.dp.meta_val_loader)
                new_val_batch = next(self.meta_val_loader)
                self.fixed_val_batches = [new_val_batch]

        if self.simulation_mode:
            total_epochs = self.config.get('epochs', 100)
            phase_switch_epoch = self.config.get('phase_switch_epoch', 8)
            peak_lambda = self.config.get('peak_lambda', 0.8)
            decay_start = int(self.config.get('decay_start', 10))
            decay_end = int(self.config.get('decay_end', 20))
            min_lambda = self.config.get('min_lambda', 0.05)
            if epoch < phase_switch_epoch:
                self.grad_reverse_lambda = 0.0
            elif epoch < decay_start:
                self.grad_reverse_lambda = peak_lambda
            else:
                t = (epoch - decay_start) / (decay_end - decay_start)
                t = min(1.0, max(0.0, t))
                self.grad_reverse_lambda = peak_lambda - t * (peak_lambda - min_lambda)
            if epoch == phase_switch_epoch:
                print(f"[Phase Transition] Switching λ from 0 to {peak_lambda} at epoch {epoch}")

        if not self.disable_lookahead and self.meta_active:
            total_epochs = self.config.get('epochs', 400)
            if epoch > 0.8 * total_epochs:
                new_eps = min(self.proactive_hvp_eps_original * 2, 1e-3)
                if self.proactive_hvp_eps < new_eps:
                    self.proactive_hvp_eps = new_eps
                    print(f"[Dynamic] Increased proactive_hvp_eps to {self.proactive_hvp_eps:.2e} at epoch {epoch}")

        if self.lookahead_switch_epoch is not None:
            self.current_disable_lookahead = (
                not self.disable_lookahead_original) if epoch >= self.lookahead_switch_epoch else self.disable_lookahead_original
        else:
            self.current_disable_lookahead = self.disable_lookahead

        total_loss = 0
        pbar = tqdm(self.dp.train_loader, desc=f"Epoch {epoch}", leave=False)
        accumulation_counter = 0

        for batch_idx, batch in enumerate(pbar):
            if accumulation_counter == 0:
                self.optimizer_model.zero_grad()

            if mode == 'fixed_weights':
                losses, _, _ = self._compute_losses_and_grads(batch)
                fixed_weights = self.config.get('fixed_weights', [1.0, 0.1, 0.05, 0.02])
                fixed_weights = torch.tensor(fixed_weights, device=self.device)
                num_losses = len(losses)
                if len(fixed_weights) > num_losses:
                    fixed_weights = fixed_weights[:num_losses]
                elif len(fixed_weights) < num_losses:
                    padding = torch.zeros(num_losses - len(fixed_weights), device=self.device)
                    fixed_weights = torch.cat([fixed_weights, padding])
                model_loss = torch.sum(fixed_weights * losses)
                final_loss = model_loss
                pbar.set_postfix({'Phase': 'Fixed', 'wM': f"{fixed_weights[0].item():.2f}"})
                main_loss, aux_loss = losses[0], losses[1]
                g_primary = torch.autograd.grad(main_loss, self.model.parameters(), retain_graph=True,
                                                allow_unused=True)
                g_primary = [g if g is not None else torch.zeros_like(p) for g, p in
                             zip(g_primary, self.model.parameters())]
                g_aux_raw = torch.autograd.grad(aux_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
                g_aux_raw = [g if g is not None else torch.zeros_like(p) for g, p in
                             zip(g_aux_raw, self.model.parameters())]
                flat_pri = torch.cat([g.view(-1) for g in g_primary])
                flat_aux = torch.cat([g.view(-1) for g in g_aux_raw])
                interference = -torch.dot(flat_pri, flat_aux).item()
                cos_sim = torch.dot(flat_pri, flat_aux) / (flat_pri.norm() * flat_aux.norm() + 1e-8)
                sr = 0.0
                if self.hessian_tracker is not None:
                    sr = self.hessian_tracker.step(model_loss, epoch, batch_idx)
                    if sr is not None:
                        print(f"[Hessian] Epoch {epoch:3d} Batch {batch_idx:4d} | λ_max={sr:.4e}")
                    else:
                        sr = 0.0
                self.last_sr = sr
                if self.grad_accumulation_steps > 1:
                    (final_loss / self.grad_accumulation_steps).backward()
                    accumulation_counter += 1
                    if accumulation_counter % self.grad_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer_model.step()
                        self.update_ema()
                        accumulation_counter = 0
                else:
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer_model.step()
                    self.update_ema()
                if hasattr(self, 'simulation_metrics'):
                    self.simulation_metrics['epoch'].append(epoch)
                    self.simulation_metrics['batch'].append(batch_idx)
                    self.simulation_metrics['interference_energy'].append(interference)
                    self.simulation_metrics['grad_cos_sim'].append(cos_sim.item())
                    self.simulation_metrics['w_aux'].append(0.0)
                    self.simulation_metrics['w_main'].append(0.0)
                    self.simulation_metrics['spectral_radius'].append(sr)
                total_loss += main_loss.item()
                continue

            if self.simulation_mode and mode == 'meta':
                rho_final = self.config.get('rho_target', -0.99)
                total_epochs = self.config.get('epochs', 100)
                warmup_end = int(total_epochs * self.config.get('conflict_warmup_ratio', 0.3))
                if epoch < warmup_end:
                    progress = epoch / max(warmup_end, 1)
                    self.rho_target = -0.1 + progress * (rho_final - (-0.1))
                else:
                    self.rho_target = rho_final
                conflict_warmup = int(total_epochs * 0.3)
                if epoch < conflict_warmup:
                    self.config['conflict_scale'] = 0.3 + (epoch / conflict_warmup) * 0.7
                else:
                    self.config['conflict_scale'] = 1.0
                self.grad_force_enabled = True
                if batch_idx == 0 and epoch % 10 == 0:
                    print(f"[Simulation] Epoch {epoch} | rho_target={self.rho_target:.3f}")
                if not hasattr(self, '_initial_main_loss'):
                    self._initial_main_loss = None
                    self._last_params_for_rollback = None
                losses, _, _ = self._compute_losses_and_grads(batch)
                main_loss, aux_loss, aux_loss2, cl_loss = losses[0], losses[1], losses[2], losses[3]
                g_primary = torch.autograd.grad(main_loss, self.model.parameters(), retain_graph=True,
                                                allow_unused=True)
                g_primary = [g if g is not None else torch.zeros_like(p) for g, p in
                             zip(g_primary, self.model.parameters())]
                g_aux_raw = torch.autograd.grad(aux_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
                g_aux_raw = [g if g is not None else torch.zeros_like(p) for g, p in
                             zip(g_aux_raw, self.model.parameters())]
                flat_pri = torch.cat([g.view(-1) for g in g_primary])
                flat_aux = torch.cat([g.view(-1) for g in g_aux_raw])
                interference = -torch.dot(flat_pri, flat_aux).item()
                cos_sim = torch.dot(flat_pri, flat_aux) / (flat_pri.norm() * flat_aux.norm() + 1e-8)
                raw_losses = losses.detach()
                interference_scaled = torch.tensor([min(abs(interference) * 1000, 10.0)], device=self.device)
                state = torch.cat([raw_losses, interference_scaled])
                _, weights_logits = self.imnet(state)
                weights_softmax = torch.softmax(weights_logits, dim=0)
                w_main_raw = weights_softmax[0]
                w_aux_raw = weights_softmax[1]
                w2_raw = weights_softmax[2]
                w3_raw = weights_softmax[3]
                w_main = torch.clamp(w_main_raw, min=0.1)
                w_aux = w_aux_raw
                w2 = w2_raw
                w3 = w3_raw
                total = w_main + w_aux + w2 + w3
                weights = torch.stack([w_main, w_aux, w2, w3]) / total

                # ========= 新增：应用权重裁剪 =========
                weights = self._clip_weights(weights)

                w_aux = weights[1].item()
                w_main = weights[0].item()
                total_loss_batch = torch.sum(weights * losses)
                sr = 0.0
                if self.hessian_tracker is not None:
                    sr = self.hessian_tracker.step(total_loss_batch, epoch, batch_idx)
                    sr = sr if sr is not None else 0.0
                self.last_sr = sr
                self.optimizer_model.zero_grad()
                self.optimizer_meta.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), 0.5)
                self.optimizer_model.step()
                self.update_ema()
                if batch_idx % self.meta_update_freq == 0:
                    self.optimizer_meta.step()
                if self._initial_main_loss is None:
                    self._initial_main_loss = main_loss.detach()
                    self._last_params_for_rollback = copy.deepcopy(self.model.state_dict())
                else:
                    if main_loss.item() > self._initial_main_loss.item() * 50.0:
                        print(f"[Warning] Loss explosion! main_loss={main_loss.item():.4f}. Rolling back.")
                        self.model.load_state_dict(self._last_params_for_rollback)
                        for param_group in self.optimizer_model.param_groups:
                            param_group['lr'] *= 0.9
                        continue
                    else:
                        self._last_params_for_rollback = copy.deepcopy(self.model.state_dict())
                if not self.fixed_val_batches:
                    try:
                        val_batch = next(self.meta_val_loader)
                        self.fixed_val_batches = [val_batch]
                    except StopIteration:
                        if hasattr(self.dp, 'meta_val_loader') and self.dp.meta_val_loader is not None:
                            self.meta_val_loader = iter(self.dp.meta_val_loader)
                            val_batch = next(self.meta_val_loader)
                            self.fixed_val_batches = [val_batch]
                        else:
                            val_batch = next(iter(self.dp.train_loader))
                            self.fixed_val_batches = [val_batch]
                if batch_idx % (self.meta_update_freq * 5) == 0:
                    self.meta_learn_step(batch, epoch, batch_idx)
                if batch_idx % 100 == 0:
                    print(
                        f"[DIAG] Epoch {epoch:3d} | Batch {batch_idx:4d} | IE={interference:.4e} | w_main={w_main:.3f} | w_aux={w_aux:.3f} | cos_sim={cos_sim.item():.4f} | rho={self.rho_target:.3f}")
                if hasattr(self, 'simulation_metrics'):
                    self.simulation_metrics['epoch'].append(epoch)
                    self.simulation_metrics['batch'].append(batch_idx)
                    self.simulation_metrics['interference_energy'].append(interference)
                    self.simulation_metrics['grad_cos_sim'].append(cos_sim.item())
                    self.simulation_metrics['w_aux'].append(w_aux)
                    self.simulation_metrics['w_main'].append(w_main)
                    self.simulation_metrics['spectral_radius'].append(sr)
                pbar.set_postfix(
                    {'Phase': 'Sim-Meta', 'wM': f"{w_main:.3f}", 'wA': f"{w_aux:.3f}", 'ρ': f"{cos_sim.item():.3f}",
                     'rho_t': f"{self.rho_target:.2f}"})
                total_loss += main_loss.item()
                continue

            elif mode == 'meta' or mode == 'random_dynamic':
                # 动态学习率调整（保留）
                if self.is_lightgcn_meta and self.is_yelp and epoch > 30:
                    for param_group in self.optimizer_model.param_groups:
                        if param_group['lr'] > 0.0001:
                            param_group['lr'] = max(0.0001, param_group['lr'] * 0.95)

                # 获取元验证批次
                try:
                    val_batch = next(self.meta_val_loader)
                except StopIteration:
                    loader_source = self.meta_val_samples if hasattr(self, 'meta_val_samples') else self.dp.train_loader
                    self.meta_val_loader = iter(loader_source)
                    val_batch = next(self.meta_val_loader)

                # 计算原始训练损失
                train_losses_original, train_diff_orig = self._compute_losses_meta(batch)

                # Random Dynamic 分支
                if self.is_random_dynamic:
                    w_main = random.uniform(0.01, 0.99)
                    w_aux = 1.0 - w_main
                    model_loss = w_main * train_losses_original[0] + w_aux * train_losses_original[1]
                    final_loss = model_loss
                    pbar.set_postfix({'Phase': 'Random', 'wM': f"{w_main:.2f}", 'wA': f"{w_aux:.2f}"})

                    # 梯度累积处理
                    if self.grad_accumulation_steps > 1:
                        final_loss = final_loss / self.grad_accumulation_steps
                        final_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer_model.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        final_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer_model.step()
                        self.update_ema()

                    total_loss += model_loss.item()
                    continue



                # Warmup 阶段
                if not self.meta_active:
                    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                    final_loss = torch.sum(weights * train_losses_original)
                    model_loss = final_loss
                    with torch.no_grad():
                        raw_w_obs, const_w_obs = self.imnet(train_losses_original.detach())
                    pbar.set_postfix(
                        {"Phase": "Warmup", "loss": f"{final_loss.item():.4f}", "rW_M": f"{raw_w_obs[0].item():.2f}"})

                    if self.grad_accumulation_steps > 1:
                        final_loss = final_loss / self.grad_accumulation_steps
                        final_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            if self.is_amazon and self.meta_active:
                                clip_norm = 0.5
                            elif self.is_lightgcn_meta and self.is_yelp:
                                clip_norm = 1.5
                            else:
                                clip_norm = 3.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                            self.optimizer_model.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        final_loss.backward()
                        if self.is_amazon and self.meta_active:
                            clip_norm = 0.5
                        elif self.is_lightgcn_meta and self.is_yelp:
                            clip_norm = 1.5
                        else:
                            clip_norm = 3.0
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                        self.optimizer_model.step()
                        self.update_ema()

                    total_loss += model_loss.item()
                    continue

                # 元学习激活后的逻辑（保持原有，但添加EMA更新）
                if not hasattr(self, 'fixed_val_batches'):
                    self.fixed_val_batches = []
                    if hasattr(self.dp, 'meta_val_loader') and self.dp.meta_val_loader is not None:
                        for i, b in enumerate(self.dp.meta_val_loader):
                            if i >= 1:  # 5← 改这里！只用1个验证batch
                                break
                            self.fixed_val_batches.append(b)
                    else:
                        if self._train_batches_cache is None:
                            self._train_batches_cache = list(self.dp.train_loader)
                        self.fixed_val_batches = random.sample(self._train_batches_cache,
                                                               min(5, len(self._train_batches_cache)))
                    print(
                        f"[Meta] Using {len(self.fixed_val_batches)} fixed validation batches for meta-loss averaging.")

                self.meta_batch_counter += 1
                do_full_meta = (self.meta_batch_counter % self.meta_update_freq == 0)

                if self.meta_active and self.meta_intervention_steps <= 3:
                    do_full_meta = False

                if do_full_meta and not self.current_disable_lookahead:
                    # 多步前瞻式元学习（Proactive）融合 Nash-MTL
                    raw_weights, weights = self.imnet(train_losses_original.detach())

                    if (self.is_yelp or self.is_amazon) and not self.current_disable_lookahead and self.meta_active:
                        weights = torch.sigmoid(weights) * 0.9 + 0.05
                    else:
                        weights = torch.sigmoid(weights) * 0.99 + 0.005

                    # 原始验证损失（使用缓存，每val_cache_freq个batch更新一次）
                    if self.meta_batch_counter - self.cached_val_step >= self.val_cache_freq:
                        val_loss_before = 0.0
                        for v_batch in self.fixed_val_batches:  # 现在只有1个batch
                            v_losses, _ = self._compute_losses_meta(v_batch)
                            val_loss_before += v_losses[0]
                        val_loss_before /= len(self.fixed_val_batches)
                        self.cached_val_loss = val_loss_before
                        self.cached_val_step = self.meta_batch_counter
                    else:
                        val_loss_before = self.cached_val_loss

                    T = self.multi_step_lookahead
                    original_params = [p.detach().clone() for p in self.model.parameters()]
                    current_params = [p.clone() for p in self.model.parameters()]
                    cumulative_val_loss = 0.0

                    for step in range(T):
                        with torch.no_grad():
                            for param, curr in zip(self.model.parameters(), current_params):
                                param.data.copy_(curr)

                        train_losses_current, _ = self._compute_losses_meta(batch)
                        num_tasks = len(train_losses_current)

                        task_grads_flat = []
                        for i in range(num_tasks):
                            loss_i = train_losses_current[i]
                            grad_i = torch.autograd.grad(loss_i, self.model.parameters(),
                                                         retain_graph=True, allow_unused=True)
                            grad_i = [g if g is not None else torch.zeros_like(p)
                                      for g, p in zip(grad_i, self.model.parameters())]
                            flat_grad = torch.cat([g.flatten() for g in grad_i])
                            task_grads_flat.append(flat_grad)

                        combined_grad_flat, λ = self._nash_mtl_gradient(task_grads_flat, max_iter=20, lr=0.1)

                        offset = 0
                        combined_grads = []
                        for p in self.model.parameters():
                            numel = p.numel()
                            g = combined_grad_flat[offset:offset + numel].view(p.shape)
                            combined_grads.append(g)
                            offset += numel

                        grad_norm = torch.norm(combined_grad_flat, 2)
                        if self.adaptive_epsilon:
                            curvature_scale = 1.0 / (grad_norm + 1e-8)
                            curvature_scale = torch.clamp(curvature_scale, 0.1, 10.0)
                            base_eps = self.epsilon_base * curvature_scale
                            base_eps = torch.clamp(base_eps, min=1e-8, max=1e-4)
                        else:
                            if self.is_amazon and self.meta_active:
                                decay = 0.7 ** (self.meta_intervention_steps + step)
                                max_eps = 5e-7
                                effective_eps = 5e-6
                                base_eps = min(effective_eps * decay / (grad_norm + 1e-8), max_eps)
                                base_eps = torch.clamp(base_eps, min=1e-8, max=max_eps)
                            elif self.is_yelp and not self.disable_lookahead and self.meta_active:
                                decay = 0.85 ** (self.meta_intervention_steps + step)
                                max_eps = 5e-7
                                effective_eps = 5e-6
                                base_eps = min(effective_eps * decay / (grad_norm + 1e-8), max_eps)
                                base_eps = torch.clamp(base_eps, min=1e-8, max=max_eps)
                            else:
                                base_eps = min(self.proactive_hvp_eps / (grad_norm + 1e-8), 1e-5)
                                base_eps = torch.clamp(base_eps, min=1e-6, max=1e-5)
                        step_scale = 1.0 / (step + 1)
                        safe_eps = base_eps * step_scale

                        new_params = []
                        for param, grad in zip(current_params, combined_grads):
                            new_param = param - safe_eps * grad
                            new_params.append(new_param)
                        current_params = new_params

                        # 计算当前步结束后的验证损失（使用临时参数）
                        with torch.no_grad():
                            for param, curr in zip(self.model.parameters(), current_params):
                                param.data.copy_(curr)

                        # ===== 优化：使用缓存，只在必要时重新计算 =====
                        # 只在第一步和最后一步精确计算，中间步使用估算
                        if step == 0 or step == T - 1:
                            val_loss_step = 0.0
                            for v_batch in self.fixed_val_batches:  # 现在只有1个batch
                                v_losses, _ = self._compute_losses_meta(v_batch)
                                val_loss_step += v_losses[0]
                            val_loss_step /= len(self.fixed_val_batches)

                            # 缓存第一步和最后一步的损失
                            if step == 0:
                                self.cached_first_val_loss = val_loss_step
                            elif step == T - 1:
                                self.cached_last_val_loss = val_loss_step
                        else:
                            # 中间步使用线性插值估算
                            progress = step / T
                            # 如果没有最后一步的值，暂时使用第一步的值
                            if self.cached_last_val_loss is None:
                                val_loss_step = self.cached_first_val_loss
                            else:
                                val_loss_step = self.cached_first_val_loss * (
                                            1 - progress) + self.cached_last_val_loss * progress

                        cumulative_val_loss += val_loss_step

                    # 最终回滚到原始参数
                    with torch.no_grad():
                        for param, orig in zip(self.model.parameters(), original_params):
                            param.data.copy_(orig)

                    # 最终扰动后验证损失（使用优化的计算方式）
                    if T > 1 and hasattr(self, 'cached_first_val_loss') and hasattr(self, 'cached_last_val_loss'):
                        # 使用第一步和最后一步的平均
                        val_loss_after = (self.cached_first_val_loss + self.cached_last_val_loss) / 2
                    else:
                        val_loss_after = cumulative_val_loss / T


                    meta_loss = (val_loss_after - val_loss_before) / (val_loss_before + 1e-8)
                    meta_loss = torch.clamp(meta_loss, min=-1.0, max=1.0)

                    if meta_loss.item() > self.config.get('loss_explosion_threshold', 3.0):
                        print(f"[Warning] Meta loss explosion: {meta_loss.item():.2f}")
                        self.config['meta_loss_weight'] *= 0.5
                        continue

                    model_loss = torch.sum(weights * train_losses_original)

                    if not hasattr(self, 'model_loss_ema'):
                        self.model_loss_ema = model_loss.detach()
                        self.meta_loss_ema = meta_loss.detach()
                    self.model_loss_ema = 0.99 * self.model_loss_ema + 0.01 * model_loss.detach()
                    self.meta_loss_ema = 0.99 * self.meta_loss_ema + 0.01 * meta_loss.detach()
                    adaptive_weight = (self.model_loss_ema / (self.meta_loss_ema + 1e-8)).clamp(0.1, 5.0)

                    meta_warmup_epochs = self.config.get('meta_warmup_epochs', 3)
                    warmup_factor = min(1.0, self.meta_intervention_steps / meta_warmup_epochs)
                    current_meta_weight = self.proactive_meta_loss_weight * warmup_factor

                    final_loss = model_loss + current_meta_weight * adaptive_weight * meta_loss

                    pbar.set_postfix({
                        'Phase': 'Proactive-Meta+Nash',
                        'W_M': f"{weights[0].item():.2f}",
                        'W_A': f"{weights[1].item():.2f}",
                        'W_C': f"{weights[2].item():.2f}",
                        'ValLoss': f"{val_loss_after.item():.4f}"
                    })

                    self.optimizer_meta.zero_grad()
                    self.optimizer_model.zero_grad()

                    if self.grad_accumulation_steps > 1:
                        final_loss = final_loss / self.grad_accumulation_steps
                        final_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                            if self.is_amazon and self.meta_active:
                                clip_norm = 1.0
                            elif self.is_lightgcn_meta and self.is_yelp:
                                clip_norm = 1.5
                            else:
                                clip_norm = 3.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                            self.optimizer_model.step()
                            self.optimizer_meta.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        final_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                        if self.is_amazon and self.meta_active:
                            clip_norm = 1.0
                        elif self.is_lightgcn_meta and self.is_yelp:
                            clip_norm = 1.5
                        else:
                            clip_norm = 3.0
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                        self.optimizer_model.step()
                        self.optimizer_meta.step()
                        self.update_ema()

                    total_loss += model_loss.item()

                else:
                    # 快速更新分支
                    # ✅ 创建 detached 版本
                    train_losses_detached = [l.detach() for l in train_losses_original]
                    if self.current_disable_lookahead:
                        raw_weights, weights = self.imnet(train_losses_original.detach())
                        if (self.is_yelp or self.is_amazon) and not self.current_disable_lookahead and self.meta_active:
                            weights = torch.sigmoid(weights) * 0.8 + 0.1
                        else:
                            weights = torch.sigmoid(weights) * 0.99 + 0.005
                        model_loss = torch.sum(weights * train_losses_original)
                        pbar.set_postfix({
                            'Phase': 'Reactive-Meta',
                            'W_M': f"{weights[0].item():.2f}",
                            'W_A1': f"{weights[1].item():.2f}",
                            'W_A2': f"{weights[2].item():.2f}",
                            'W_C': f"{weights[3].item():.2f}",
                        })
                    else:
                        weights = self.imnet(train_losses_original.detach())[1]
                        if (self.is_yelp or self.is_amazon) and not self.current_disable_lookahead and self.meta_active:
                            weights = torch.sigmoid(weights) * 0.8 + 0.1
                        else:
                            weights = torch.sigmoid(weights) * 0.99 + 0.005
                        model_loss = torch.sum(weights * train_losses_original)
                        pbar.set_postfix({
                            'Phase': 'Fast-Meta',
                            'W_M': f"{weights[0].item():.2f}",
                            'W_A': f"{weights[1].item():.2f}",
                            'W_C': f"{weights[2].item():.2f}"
                        })

                    self.optimizer_meta.zero_grad()
                    self.optimizer_model.zero_grad()

                    if self.grad_accumulation_steps > 1:
                        model_loss = model_loss / self.grad_accumulation_steps
                        model_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                            if self.current_disable_lookahead:
                                clip_norm = 3.0
                            else:
                                clip_norm = 1.0 if self.is_amazon and self.meta_active else 3.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                            self.optimizer_model.step()
                            self.optimizer_meta.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                        if self.current_disable_lookahead:
                            clip_norm = 3.0
                        else:
                            clip_norm = 1.0 if self.is_amazon and self.meta_active else 3.0
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                        self.optimizer_model.step()
                        self.optimizer_meta.step()
                        self.update_ema()

                    total_loss += model_loss.item()


            elif mode == 'scalarization':
                losses = self._compute_losses(batch)
                weights = torch.tensor(self.config.get('static_weights', [1.0, 0.1, 0.05]), device=self.device)
                final_loss = torch.sum(weights * losses)
                pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})
                if self.grad_accumulation_steps > 1:
                    (final_loss / self.grad_accumulation_steps).backward()
                    accumulation_counter += 1
                    if accumulation_counter % self.grad_accumulation_steps == 0:
                        self.optimizer_model.step()
                        self.update_ema()
                        accumulation_counter = 0
                else:
                    final_loss.backward()
                    self.optimizer_model.step()
                    self.update_ema()
                total_loss += final_loss.item()

            elif mode == 'pcgrad':
                losses = self._compute_losses(batch)
                total_loss += self._step_pcgrad(losses, epoch)

            elif mode == 'mgda':
                losses = self._compute_losses(batch)
                total_loss += self._step_mgda(losses, epoch)

            elif mode == 'gradnorm':
                losses = self._compute_losses(batch)
                total_loss += self._step_gradnorm(losses)

            elif mode == 'single':
                if self.config.get('model_name') == 'SimGCL':
                    losses = self._compute_losses(batch)
                    cl_lambda = self.config.get('cl_lambda', 0.1)
                    final_loss = losses[0] + cl_lambda * losses[2]
                else:
                    users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]
                    if self.model_name == 'NCF':
                        u_emb, i_emb = self.model.get_all_embeddings()
                    else:
                        adj = (self.dp.norm_adj.to(self.device) if hasattr(self.dp, 'norm_adj') else None)
                        u_emb, i_emb = self.model.get_all_embeddings(adj)
                    batch_u = u_emb[users]
                    batch_pos = i_emb[pos_items]
                    batch_neg = i_emb[neg_items]
                    pos_scores = torch.sum(batch_u * batch_pos, dim=1)
                    neg_scores = torch.sum(batch_u * batch_neg, dim=1)
                    bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                    reg_loss = (1 / 2) * (
                                batch_u.norm(2).pow(2) + batch_pos.norm(2).pow(2) + batch_neg.norm(2).pow(2)) / float(
                        users.shape[0])
                    final_loss = bpr_loss + self.config.get('wd', 1e-4) * reg_loss
                pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})
                if self.grad_accumulation_steps > 1:
                    (final_loss / self.grad_accumulation_steps).backward()
                    accumulation_counter += 1
                    if accumulation_counter % self.grad_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        self.optimizer_model.step()
                        self.update_ema()
                        accumulation_counter = 0
                else:
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer_model.step()
                    self.update_ema()
                total_loss += final_loss.item()

        if accumulation_counter > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            self.optimizer_model.step()
            self.update_ema()
        return total_loss / len(self.dp.train_loader)

    def update_lr(self, current_ndcg, epoch):
        if self.simulation_mode:
            if current_ndcg > self.best_metric:
                self.best_metric = current_ndcg
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.last_best_update_epoch = epoch
            if not self.meta_active and epoch - self.last_best_update_epoch >= self.force_meta_patience:
                print("[Simulation] Forcing meta-learning intervention due to stagnation.")
                self.meta_active = True
                self.meta_intervention_steps = 0
                self.pre_intervention_state = copy.deepcopy(self.model.state_dict())
                for param_group in self.optimizer_model.param_groups:
                    param_group['lr'] *= 0.8
            if self.meta_cool_down > 0:
                self.meta_cool_down -= 1
            return
        # 其余调度逻辑与原代码一致（略）

    def stable_conflict_index(self, g_p_list, g_a_list, ema=None, alpha=0.95):
        flat_p = torch.cat([g.view(-1) for g in g_p_list if g is not None])
        flat_a = torch.cat([g.view(-1) for g in g_a_list if g is not None])
        dot_pp = torch.dot(flat_p, flat_p)
        dot_aa = torch.dot(flat_a, flat_a)
        dot_pa = torch.dot(flat_p, flat_a)
        denom = dot_pp + dot_aa + 1e-8
        c_raw = 1.0 - (dot_pp + 2 * dot_pa + dot_aa) / denom
        c_raw = torch.clamp(c_raw, 0.0, 1.0)
        if ema is None:
            return c_raw, c_raw
        else:
            c_ema = alpha * ema + (1 - alpha) * c_raw
            return c_ema, c_ema

    def virtual_update(self, grad_primary, grad_aux, w_main, w_aux, lr=None, noise_scale=0.0):
        if lr is None:
            lr = self.inner_lr
        new_params = []
        with torch.no_grad():
            for p, gp, ga in zip(self.model.parameters(), grad_primary, grad_aux):
                if gp is None: gp = torch.zeros_like(p)
                if ga is None: ga = torch.zeros_like(p)
                delta = w_main * gp + w_aux * ga
                if noise_scale > 0:
                    delta += torch.randn_like(delta) * noise_scale * p.norm()
                new_p = p - lr * delta
                new_params.append(new_p.clone())
        return new_params

    def meta_learn_step(self, batch, epoch, batch_idx):
        losses, _, _ = self._compute_losses_and_grads(batch)
        main_loss, aux_loss = losses[0], losses[1]
        g_primary = torch.autograd.grad(main_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        g_primary = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_primary, self.model.parameters())]
        g_aux = torch.autograd.grad(aux_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        g_aux = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_aux, self.model.parameters())]
        c, self.conflict_ema = self.stable_conflict_index(g_primary, g_aux, self.conflict_ema, alpha=0.95)
        raw_losses = losses.detach()
        flat_pri = torch.cat([g.view(-1) for g in g_primary])
        flat_aux = torch.cat([g.view(-1) for g in g_aux])
        interference = -torch.dot(flat_pri, flat_aux).item()
        interference_scaled = torch.tensor([interference * 200000], device=self.device)
        state = torch.cat([raw_losses, interference_scaled])
        _, weights_logits = self.imnet(state)
        weights_softmax = torch.softmax(weights_logits, dim=0)
        w_sum = weights_softmax[0] + weights_softmax[1] + 1e-8
        w_main = weights_softmax[0] / w_sum
        w_aux = weights_softmax[1] / w_sum
        w_aux_raw = w_aux

        if not hasattr(self, 'w_aux_ema'):
            self.w_aux_ema = w_aux.item()
        self.w_aux_ema = 0.95 * self.w_aux_ema + 0.05 * w_aux.item()
        w_aux_for_update = w_aux_raw.item()
        num_val_batches = min(5, len(self.fixed_val_batches))
        if num_val_batches == 0:
            return 0.0
        val_losses_before = []
        val_losses_after = []
        for i in range(num_val_batches):
            v_batch = self.fixed_val_batches[i]
            with torch.no_grad():
                v_losses_before, _, _ = self._compute_losses_and_grads(v_batch)
                val_loss_before = v_losses_before[0].detach()
                val_losses_before.append(val_loss_before)
            virtual_params = self.virtual_update(g_primary, g_aux, w_main.item(), w_aux_for_update, lr=self.inner_lr)
            original_state = [p.clone() for p in self.model.parameters()]
            with torch.no_grad():
                for p, vp in zip(self.model.parameters(), virtual_params):
                    p.copy_(vp)
            with torch.no_grad():
                v_losses_after, _, _ = self._compute_losses_and_grads(v_batch)
                val_loss_after = v_losses_after[0].detach()
                val_losses_after.append(val_loss_after)
            with torch.no_grad():
                for p, orig in zip(self.model.parameters(), original_state):
                    p.copy_(orig)
        val_loss_before = torch.stack(val_losses_before).mean()
        val_loss_after = torch.stack(val_losses_after).mean()
        delta_val = (val_loss_before - val_loss_after) / (val_loss_before.abs() + 1e-8)

        target_w = 0.05  # 辅助权重目标值降至 0.05
        lambda_quad_effective = self.config.get('lambda_quad', 1.0)
        if abs(w_aux_raw - target_w) > 0.05:
            quad_penalty = lambda_quad_effective * ((w_aux_raw - target_w) ** 2)
        else:
            quad_penalty = torch.tensor(0.0, device=self.device)
        sparsity_reg = -self.lambda_sparsity * torch.log(w_aux_raw + 1e-6)
        if epoch < 8:
            effective_lambda_interference = self.lambda_interference * 0.1
        else:
            effective_lambda_interference = self.lambda_interference
        if w_aux_raw > 0.2:
            interference_penalty = effective_lambda_interference * abs(interference) * w_aux_raw
        else:
            interference_penalty = torch.tensor(0.0, device=self.device)
        spectral_penalty = torch.tensor(0.0, device=self.device)
        if (hasattr(self, 'hessian_tracker') and self.hessian_tracker is not None
                and hasattr(self, 'lambda_spectral') and self.lambda_spectral > 0):
            spectral_radius = getattr(self, 'last_sr', 0.0)
            if epoch < 8:
                effective_lambda_spectral = self.lambda_spectral * 0.1
            else:
                effective_lambda_spectral = self.lambda_spectral
            spectral_penalty = torch.tensor(effective_lambda_spectral * max(0.0, spectral_radius - 1.0),
                                            device=self.device)

        # 元损失仅由正则项组成，不再使用 -delta_val 和熵正则
        meta_loss = quad_penalty + sparsity_reg + interference_penalty + spectral_penalty

        self.optimizer_meta.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
        self.optimizer_meta.step()
        if batch_idx % 50 == 0:
            phase = "PhaseI(探索)" if epoch < 8 else "PhaseII(抑制)"
            print(
                f"[MetaStep] E{epoch:3d} B{batch_idx:4d} | {phase} | c={c:.4f} | w_aux={w_aux.item():.4f} | delta_val={delta_val.item():.4f} | meta={meta_loss.item():.4f} | interference={interference:.6f} | spectral={spectral_penalty.item():.6f} | val_loss={val_loss_after.item():.6f}")
        return meta_loss.item()

    def compute_loss_with_virtual_params(self, batch, virtual_params):
        original_state = [p.clone() for p in self.model.parameters()]
        with torch.no_grad():
            for p, vp in zip(self.model.parameters(), virtual_params):
                p.copy_(vp)
        if self.simulation_mode:
            losses, _, _ = self._compute_losses_and_grads(batch)
            main_loss = losses[0].detach()
        else:
            losses, _ = self._compute_losses_meta(batch)
            main_loss = losses[0].detach()
        with torch.no_grad():
            for p, orig in zip(self.model.parameters(), original_state):
                p.copy_(orig)
        return main_loss

    def compute_meta_loss(self, val_loss, spectral_radius=None):
        meta_loss = val_loss
        if spectral_radius is not None and self.lambda_interference > 0:
            current_epoch = getattr(self, 'current_epoch', 0)
            if current_epoch < 8:
                effective_lambda = self.lambda_interference * 0.1
            else:
                effective_lambda = self.lambda_interference
            if not hasattr(self, 'beta'):
                self.beta = 1.0
            interference_energy = self.beta * spectral_radius
            interference_penalty = effective_lambda * torch.relu(
                torch.tensor(interference_energy, device=val_loss.device) - self.interference_threshold
            )
            meta_loss = meta_loss + interference_penalty
        return meta_loss

    def _get_random_weights(self):
        weights = torch.rand(3, device=self.device)
        return weights / (weights.sum() + 1e-8)

    def train(self):
        epochs = self.config.get('epochs', 200)
        top_k = self.config.get('top_k', 20)
        eval_step = self.config.get('eval_freq', 1)
        best_recall, best_ndcg, best_epoch = 0.0, 0.0, 0
        print(f"[Trainer Info] Start training {self.model_name} on {self.config.get('dataset')} in '{self.mode}' mode for {epochs} epochs.")
        loss_window = []
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch=epoch, mode=self.mode)
            loss_window.append(loss)
            if len(loss_window) > 5:
                loss_window.pop(0)
            avg_loss = np.mean(loss_window) if loss_window else loss
            if epoch % eval_step == 0 or epoch == 1:
                old_best_metric = self.best_metric
                if self.use_ema and epoch > self.warmup_epochs:
                    recall, ndcg = self.evaluate_with_ema(top_k=top_k)
                else:
                    recall, ndcg = self.evaluate(top_k=top_k)
                self.update_lr(ndcg, epoch)
                if self.simulation_mode and not self.meta_active:
                    if ndcg <= old_best_metric:
                        self.sim_patience_counter += 1
                        if self.sim_patience_counter >= 5:
                            for param_group in self.optimizer_model.param_groups:
                                param_group['lr'] *= 0.8
                            print(f"[Simulation] Plateau detected, reducing LR to {param_group['lr']:.6f}")
                            self.sim_patience_counter = 0
                    else:
                        self.sim_patience_counter = 0
                if self.meta_active:
                    self.meta_intervention_steps += 1
                    if self.avg_loss_before_meta is not None:
                        if self.is_amazon:
                            if loss < self.avg_loss_before_meta * 0.7:
                                self.avg_loss_before_meta = 0.98 * self.avg_loss_before_meta + 0.02 * loss
                            explosion_ratio = self.proactive_loss_explosion_ratio
                        elif self.is_yelp and not self.disable_lookahead and self.meta_active:
                            if loss < self.avg_loss_before_meta * 0.7:
                                self.avg_loss_before_meta = 0.98 * self.avg_loss_before_meta + 0.02 * loss
                            explosion_ratio = self.proactive_loss_explosion_ratio
                        else:
                            if loss < self.avg_loss_before_meta * 0.9:
                                self.avg_loss_before_meta = 0.9 * self.avg_loss_before_meta + 0.1 * loss
                            explosion_ratio = self.proactive_loss_explosion_ratio
                        if loss > self.avg_loss_before_meta * explosion_ratio:
                            print(f"[Meta] Loss explosion detected (current={loss:.4f}, baseline={self.avg_loss_before_meta:.4f}, ratio={explosion_ratio:.1f}x). Immediate rollback.")
                            self.model.load_state_dict(self.pre_intervention_state)
                            self.meta_active = False
                            self.meta_cool_down = 15
                            print(f"[Meta] Cooling down for {self.meta_cool_down} epochs.")
                            self.avg_loss_before_meta = None
                            if ndcg > best_ndcg:
                                best_recall, best_ndcg, best_epoch = recall, ndcg, epoch
                                print(f"[Best Updated] Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {best_recall:.4f} | N@{top_k}: {best_ndcg:.4f}  <-- New Best!")
                            else:
                                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {recall:.4f} | N@{top_k}: {ndcg:.4f} | (Best N: {best_ndcg:.4f})")
                            continue
                    if ndcg > self.best_metric:
                        self.best_metric = ndcg
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.consecutive_decreases = 0
                        print(f"[Meta] Intervention effective: new best NDCG = {ndcg:.4f}")
                    if self.meta_intervention_steps >= self.meta_max_steps:
                        if ndcg <= self.best_metric:
                            print(f"[Meta] No improvement after {self.meta_max_steps} steps. Rolling back.")
                            self.model.load_state_dict(self.pre_intervention_state)
                        else:
                            print(f"[Meta] Intervention succeeded! Keeping improved model.")
                        self.meta_active = False
                        self.meta_cool_down = 8
                        self.avg_loss_before_meta = None
                        print(f"[Meta] Cooling down for {self.meta_cool_down} epochs.")
                if ndcg > best_ndcg:
                    best_recall, best_ndcg, best_epoch = recall, ndcg, epoch
                    print(f"[Best Updated] Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {best_recall:.4f} | N@{top_k}: {best_ndcg:.4f}  <-- New Best!")
                    if self.early_stop_patience > 0:
                        self.no_improve_epochs = 0
                else:
                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {recall:.4f} | N@{top_k}: {ndcg:.4f} | (Best N: {best_ndcg:.4f})")
                    if self.early_stop_patience > 0:
                        self.no_improve_epochs += 1
                        if self.no_improve_epochs >= self.early_stop_patience:
                            print(f"[Early Stop] No improvement for {self.early_stop_patience} epochs. Stopping training.")
                            break
            else:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | [Eval Skipped]")
            if (self.meta_active and self.meta_intervention_steps == 1 and self.avg_loss_before_meta is None):
                baseline_loss = min(loss_window) if loss_window else avg_loss
                self.avg_loss_before_meta = baseline_loss
                print(f"[Meta] Recorded baseline loss for explosion detection: {self.avg_loss_before_meta:.4f} (min of recent {len(loss_window)} epochs)")
        return best_recall, best_ndcg

    # 以下辅助方法保持不变
    def _step_pcgrad(self, losses, epoch):
        import random
        decay = 1.0 if epoch <= 40 else max(0.0, 1.0 - (epoch - 40) / 40.0)
        base_weights = torch.tensor([1.0, 0.1 * decay, 0.05 * decay], device=self.device)
        weighted_losses = [losses[i] * base_weights[i] for i in range(len(losses))]
        num_tasks = len(weighted_losses)
        task_grads = []
        for i in range(num_tasks):
            self.optimizer_model.zero_grad()
            weighted_losses[i].backward(retain_graph=True)
            grad_vec = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1))
                else:
                    grad_vec.append(torch.zeros_like(param).view(-1))
            task_grads.append(torch.cat(grad_vec))
        task_order = list(range(num_tasks))
        random.shuffle(task_order)
        modified_grads = [task_grads[i].clone() for i in range(num_tasks)]
        for i in task_order:
            for j in task_order:
                if i == j: continue
                dot_prod = torch.dot(modified_grads[i], task_grads[j])
                if dot_prod < 0:
                    denominator = torch.dot(task_grads[j], task_grads[j])
                    if denominator > 1e-6: modified_grads[i] -= (dot_prod / denominator) * task_grads[j]
        total_grad = sum(modified_grads)
        self.optimizer_model.zero_grad()
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param.grad = total_grad[offset:offset + numel].view(param.shape).clone()
                offset += numel
        self.optimizer_model.step()
        self.update_ema()
        return sum([l.item() for l in weighted_losses])

    def _step_mgda(self, losses, epoch):
        decay = 1.0 if epoch <= 20 else max(0.0, 1.0 - (epoch - 20) / 30.0)
        base_weights = torch.tensor([1.0, 0.1 * decay, 0.05 * decay], device=self.device)
        task_grads = []
        for i in range(len(losses)):
            self.optimizer_model.zero_grad()
            (losses[i] * base_weights[i]).backward(retain_graph=True)
            grads = []
            for param in self.model.parameters():
                if param.grad is not None: grads.append(param.grad.view(-1))
            if grads:
                task_grads.append(torch.cat(grads))
            else:
                task_grads.append(torch.zeros(1, device=self.device))
        if len(task_grads) < 2:
            self.optimizer_model.step()
            self.update_ema()
            return sum([l.item() for l in losses])
        num_tasks = len(task_grads)
        main_grad = task_grads[0]
        aux_grad = sum(task_grads[1:]) / (num_tasks - 1)
        v = aux_grad - main_grad
        dist_sq = torch.dot(v, v)
        alpha = torch.clamp(torch.dot(v, aux_grad) / dist_sq, 0.01, 0.99) if dist_sq > 1e-8 else 0.5
        final_grad_vec = alpha * main_grad + (1 - alpha) * aux_grad
        self.optimizer_model.zero_grad()
        offset = 0
        for param in self.model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param.grad = final_grad_vec[offset:offset + numel].view(param.shape).clone()
                offset += numel
        self.optimizer_model.step()
        self.update_ema()
        return (alpha * (losses[0] * base_weights[0]) + (1 - alpha) * sum(
            [losses[i] * base_weights[i] for i in range(1, num_tasks)])).item()

    # FIXED: 添加缺失的 _step_gradnorm 初始化
    def _step_gradnorm(self, losses):
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in losses]
            self.gradnorm_weights = torch.nn.Parameter(
                torch.ones(len(losses), device=self.device) / len(losses)
            )
            self.gradnorm_optimizer = optim.AdamW([self.gradnorm_weights], lr=0.01)

        num_tasks = len(losses)

        # ─────────────────────────────────────────────────────────────
        # Step 1: 在 optimizer_model.step() 之前，基于当前参数计算
        #         每个任务的梯度范数（需要 create_graph=True 以便后续
        #         对 gradnorm_weights 求梯度）。
        #         此时 losses[i] 的计算图仍指向当前（未更新）参数，
        #         shared_weight 也是当前参数，两者一致。
        # ─────────────────────────────────────────────────────────────
        if hasattr(self.model, 'user_embedding'):
            shared_weight = self.model.user_embedding.weight
        elif hasattr(self.model, 'embedding_user'):
            shared_weight = self.model.embedding_user.weight
        else:
            shared_weight = next(self.model.parameters())
            print(
                "[Warning] Could not find user_embedding or embedding_user, "
                "using first parameter as shared weight."
            )

        grad_norms = []
        for i in range(num_tasks):
            # gradnorm_weights[i] 参与梯度（不 detach），
            # create_graph=True 保留二阶图供 grad_loss.backward() 使用
            loss_i = self.gradnorm_weights[i] * losses[i]
            grad = torch.autograd.grad(
                loss_i, shared_weight,
                retain_graph=True,   # 保留图供后续任务及 total_loss.backward 使用
                create_graph=True    # 保留二阶图供 grad_loss.backward 使用
            )[0]
            grad_norms.append(torch.norm(grad, 2))

        grad_norms = torch.stack(grad_norms)

        # ─────────────────────────────────────────────────────────────
        # Step 2: 计算 GradNorm 目标值（targets），不需要梯度
        # ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            mean_norm = grad_norms.detach().mean()
            loss_ratios = torch.tensor(
                [losses[i].item() / (self.initial_losses[i] + 1e-8)
                 for i in range(num_tasks)],
                device=self.device
            )
            inverse_train_rate = loss_ratios / loss_ratios.mean()
            targets = mean_norm * (inverse_train_rate ** 1.0)

        # ─────────────────────────────────────────────────────────────
        # Step 3: 更新 gradnorm_weights（在 optimizer_model.step 之前）
        #         grad_loss 依赖 grad_norms → 依赖 gradnorm_weights，
        #         此时计算图完整，backward 正确。
        # ─────────────────────────────────────────────────────────────
        grad_loss = torch.sum(torch.abs(grad_norms - targets))

        self.gradnorm_optimizer.zero_grad()
        grad_loss.backward(retain_graph=True)   # 保留图供下方 total_loss.backward 使用
        self.gradnorm_optimizer.step()

        # 重新归一化权重，使其之和等于任务数
        with torch.no_grad():
            self.gradnorm_weights.mul_(
                len(losses) / torch.sum(self.gradnorm_weights)
            )

        # ─────────────────────────────────────────────────────────────
        # Step 4: 使用 detach 后的权重计算加权总损失，
        #         更新主模型参数（此步骤不涉及 gradnorm_weights 梯度）
        # ─────────────────────────────────────────────────────────────
        weighted_losses = [
            self.gradnorm_weights[i].detach() * losses[i]
            for i in range(num_tasks)
        ]
        total_loss = sum(weighted_losses)

        self.optimizer_model.zero_grad()
        total_loss.backward()   # 计算图已在 Step3 backward 后保留，此处正常释放
        if 'grad_clip_norm' in self.config:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config['grad_clip_norm']
            )
        self.optimizer_model.step()

        # ─────────────────────────────────────────────────────────────
        # Step 5: EMA 更新（在主模型参数更新后执行）
        # ─────────────────────────────────────────────────────────────
        self.update_ema()

        return total_loss.item()