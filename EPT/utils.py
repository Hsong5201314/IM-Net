"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability
in multi-objective learning via spectral regularization
File: utils.py
Description:
    High-performance scientific utilities for spectral analysis of the loss
    landscape. Implements matrix-free Power Iteration for Hessian spectral
    radius estimation, supporting the verification of dynamical stability.
    [Enhanced] Added gradient conflict metrics for simulation mode:
        - interference_energy: - (g_pri · g_aux)
        - gradient_cosine_similarity: cos(g_pri, g_aux)
Date: 2026-05-03
"""

import torch
import numpy as np


def compute_hessian_spectral_radius(model, loss, num_iter=10, tolerance=1e-6,
                                    verbose=False, loss_scaling=1.0):
    """
    Quantifies the maximum eigenvalue (lambda_max) of the Hessian matrix.

    Mathematical Basis:
    Matrix-free Power Iteration via Hessian-Vector Product (HVP).
    This computes the 'sharpness' of the optimization manifold, enabling
    the Proactive Meta-Control (IM-Net) to regularize spectral instability.

    Args:
        model: The neural manifold encoder (e.g., LightGCN).
        loss: The scalar energy functional (Weighted Loss).
              !! 调用方必须保证此 loss 的计算图尚未被 backward() 释放，
                 且 loss 通过 model.parameters() 可微。
        num_iter: Iterations for spectral convergence.
        tolerance: Numerical stability epsilon.
        verbose: If True, prints intermediate norms.
        loss_scaling: Multiplicative factor for loss (helps stability when loss is large).

    Returns:
        float: The spectral radius (maximum curvature) of the Hessian.
               Returns 0.0 if computation fails for any reason.
    """
    if loss_scaling != 1.0:
        loss = loss * loss_scaling

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0

    # ── 修复1：不再对 loss 做 detach().requires_grad_(True)。
    #    原代码在 HessianTracker.step() 中执行该操作，切断了 loss 与
    #    model.parameters() 之间的计算图，使 autograd.grad(loss, params)
    #    无法找到路径，触发 RuntimeError。
    #    正确做法：直接使用调用方传入的、计算图完整的 loss。
    try:
        # 1. Compute first-order gradient field（需要 create_graph=True 以支持 HVP）
        grads = torch.autograd.grad(
            loss, params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)
        ]
    except RuntimeError as e:
        # 计算图已释放或其他原因导致失败，安全降级返回 0.0
        if verbose:
            print(f"[HessianTracker] grad computation failed: {e}")
        return 0.0

    def flatten_tensors(tensors):
        return torch.cat([t.contiguous().view(-1) for t in tensors])

    flat_grads = flatten_tensors(grads)

    # 3. Initialize curvature probing vector
    v = torch.randn(flat_grads.size(), device=flat_grads.device)
    v = v / (torch.norm(v) + tolerance)

    # 4. Iterative Spectral Refinement
    try:
        for it in range(num_iter):
            gv_product = torch.sum(flat_grads * v)
            hvp_tensors = torch.autograd.grad(
                gv_product, params,
                retain_graph=True,
                allow_unused=True
            )
            hvp = flatten_tensors([
                h if h is not None else torch.zeros_like(p)
                for h, p in zip(hvp_tensors, params)
            ])
            v_new = hvp / (torch.norm(hvp) + tolerance)
            if verbose:
                diff = torch.norm(v_new - v).item()
                print(f"  HVP iteration {it + 1}: diff={diff:.4e}")
            v = v_new.detach()

        # 5. Rayleigh Quotient: λ_max ≈ v^T H v
        gv_product = torch.sum(flat_grads * v)
        hvp_tensors = torch.autograd.grad(
            gv_product, params,
            retain_graph=True,
            allow_unused=True
        )
        hvp = flatten_tensors([
            h if h is not None else torch.zeros_like(p)
            for h, p in zip(hvp_tensors, params)
        ])
        spectral_radius = torch.dot(v, hvp).item()

    except RuntimeError as e:
        if verbose:
            print(f"[HessianTracker] HVP iteration failed: {e}")
        return 0.0
    finally:
        # 6. 显式释放中间变量（不影响调用方的计算图）
        del flat_grads

    return abs(spectral_radius)


def compute_gradient_conflict_metrics(g_primary, g_aux):
    """
    Compute Interference Energy and Cosine Similarity between primary and auxiliary gradients.

    Args:
        g_primary: List of tensors (gradients of primary loss w.r.t model params)
        g_aux:     List of tensors (gradients of auxiliary loss w.r.t model params)

    Returns:
        interference_energy (float): -(g_pri · g_aux). Positive = conflict.
        cosine_similarity (float):   dot / (||g_pri|| * ||g_aux|| + eps).
    """
    flat_pri = torch.cat([g.view(-1) for g in g_primary])
    flat_aux = torch.cat([g.view(-1) for g in g_aux])
    dot      = torch.dot(flat_pri, flat_aux)
    norm_pri = torch.norm(flat_pri) + 1e-8
    norm_aux = torch.norm(flat_aux) + 1e-8
    cos_sim  = dot / (norm_pri * norm_aux)
    interference = -dot.item()
    return interference, cos_sim.item()


def compute_gradient_interference_from_losses(model, loss_primary, loss_aux,
                                              retain_graph=True):
    """
    Convenience function to compute interference energy and cosine similarity
    directly from two loss tensors.

    Args:
        model:         Model with parameters.
        loss_primary:  Scalar tensor for primary task.
        loss_aux:      Scalar tensor for auxiliary task.
        retain_graph:  Whether to keep computational graph for further backward passes.

    Returns:
        interference_energy (float)
        cosine_similarity   (float)
        gradients_primary   (list)
        gradients_aux       (list)
    """
    g_primary = torch.autograd.grad(
        loss_primary, model.parameters(),
        create_graph=False, retain_graph=retain_graph, allow_unused=True
    )
    g_primary = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(g_primary, model.parameters())
    ]
    g_aux = torch.autograd.grad(
        loss_aux, model.parameters(),
        create_graph=False, retain_graph=retain_graph, allow_unused=True
    )
    g_aux = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(g_aux, model.parameters())
    ]
    interference, cos_sim = compute_gradient_conflict_metrics(g_primary, g_aux)
    return interference, cos_sim, g_primary, g_aux


class HessianTracker:
    """
    A lightweight wrapper to compute and log the Hessian spectral radius periodically
    during training without cluttering the main loop.

    !! 重要调用约定（修复后）：
       hessian_tracker.step(loss, epoch, batch_idx) 必须在 loss.backward() 之前调用。
       原因：compute_hessian_spectral_radius 需要完整的计算图（create_graph=True），
       一旦 backward() 执行，图将被释放，后续无法再次对同一 loss 求高阶导数。

    Usage:
        total_loss = compute_loss(...)

        # Step 1: 先计算谱半径（图完整）
        sr = tracker.step(total_loss, epoch, batch_idx)

        # Step 2: 再做模型参数更新（会释放计算图）
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    """

    def __init__(self, model, compute_freq=10, num_iter=10, loss_scaling=1.0):
        self.model        = model
        self.compute_freq = compute_freq
        self.num_iter     = num_iter
        self.loss_scaling = loss_scaling
        self.step_counter = 0

    def step(self, loss, epoch, batch_idx):
        """
        Returns spectral radius (float) if this is a compute step, else None.

        !! loss 必须是计算图完整的 tensor（尚未经过 backward()）。
        """
        self.step_counter += 1
        if self.step_counter % self.compute_freq != 0:
            return None

        # ── 修复2：删除 loss.detach().requires_grad_(True)。
        #    原代码：
        #        if not loss.requires_grad:
        #            loss = loss.detach().requires_grad_(True)
        #    该操作将 loss 与 model.parameters() 的计算图完全切断，
        #    导致 compute_hessian_spectral_radius 中
        #    autograd.grad(loss, params) 报 RuntimeError。
        #
        #    修复：直接使用原始 loss（调用方负责在 backward() 前调用本方法）。
        #    如果 loss 本身不需要梯度（例如被 detach 过），则无法计算，安全返回 0.0。
        if not loss.requires_grad:
            return 0.0

        # 保持模型处于训练模式（HVP 计算需要参数梯度）
        was_training = self.model.training
        self.model.train()

        with torch.enable_grad():
            sr = compute_hessian_spectral_radius(
                self.model, loss,
                num_iter=self.num_iter,
                loss_scaling=self.loss_scaling
            )

        # 恢复原始训练模式
        if not was_training:
            self.model.eval()

        return sr


def fix_random_seed(seed=2026):
    """
    Ensures deterministic behavior across stochastic tensor operations.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[*] Deterministic seed locked: {seed}")
