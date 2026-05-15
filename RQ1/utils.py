"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: utils.py
Description:
    High-performance scientific utilities for spectral analysis of the loss
    landscape. Implements matrix-free Power Iteration for Hessian spectral
    radius estimation, supporting the verification of dynamical stability.
Author: Song HUANG (huangsong@cque.edu.cn)
Date: 2026-02-14
"""

import torch
import numpy as np

def compute_hessian_spectral_radius(model, loss, num_iter=10, tolerance=1e-6):
    """
    Quantifies the maximum eigenvalue (lambda_max) of the Hessian matrix.

    Mathematical Basis:
    Matrix-free Power Iteration via Hessian-Vector Product (HVP).
    This computes the 'sharpness' of the optimization manifold, enabling
    the Proactive Meta-Control (IM-Net) to regularize spectral instability.

    Args:
        model: The neural manifold encoder (e.g., LightGCN).
        loss: The scalar energy functional (Weighted Loss).
        num_iter: Iterations for spectral convergence.

    Returns:
        float: The spectral radius (maximum curvature) of the Hessian.
    """
    # 1. Isolate the differentiable manifold
    params = [p for p in model.parameters() if p.requires_grad]

    # 2. Compute first-order gradient field
    # create_graph=True is essential for higher-order derivatives
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    def flatten_tensors(tensors):
        return torch.cat([t.contiguous().view(-1) for t in tensors])

    flat_grads = flatten_tensors(grads)

    # 3. Initialize curvature probing vector (v)
    # Start with a random unit vector on the parameter sphere
    v = torch.randn(flat_grads.size()).to(flat_grads.device)
    v = v / (torch.norm(v) + tolerance)

    # 4. Iterative Spectral Refinement
    for _ in range(num_iter):
        # Implicit Hessian-Vector Product: H * v = ∇(∇L · v)
        gv_product = torch.sum(flat_grads * v)

        # Second-order differentiation
        hvp_tensors = torch.autograd.grad(gv_product, params, retain_graph=True)
        hvp = flatten_tensors(hvp_tensors)

        # Update and normalize the principal eigenvector
        v = hvp / (torch.norm(hvp) + tolerance)

        # Crucial for memory: detach the eigenvector from the graph
        v = v.detach()

    # 5. Rayleigh Quotient: λ_max ≈ (v.T * H * v)
    # Perform one final HVP for the converged v
    gv_product = torch.sum(flat_grads * v)
    hvp_tensors = torch.autograd.grad(gv_product, params, retain_graph=True)
    hvp = flatten_tensors(hvp_tensors)

    spectral_radius = torch.dot(v, hvp).item()

    # 6. Explicit Memory Reclamation
    # To prevent OOM during long training loops in Multi-Objective runs
    del flat_grads
    del hvp

    return abs(spectral_radius)

def fix_random_seed(seed=2026):
    """
    Ensures deterministic behavior across stochastic tensor operations.
    Critical for the 'Digital Twin' reproducibility protocol.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[*] Deterministic seed locked: {seed}")
