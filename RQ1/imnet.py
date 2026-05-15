"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: imnet.py
Description:
    Interference Mitigation Network (IM-Net) with Multi-Task Contrastive Enhancement.
    Implements a proactive meta-controller that penalizes high-curvature
    optimization trajectories by integrating the Hessian Spectral Radius
    into the meta-gradient flow.
Author: Song HUANG (huangsong@cque.edu.cn)
Date: 2026-03-24 (Upgraded for Top-tier Journals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IMNet(nn.Module):
    """
    The IM-Net Meta-Learner (The 'Maxwell's Demon' of Multi-Objective Optimization).

    It dynamically transforms task-specific loss features into optimal task weights.
    The network is optimized to minimize the 'Interference Energy' on the objective
    manifold, defined by the spectral radius of the Hessian.
    Supported Tasks: [Main BPR Loss, Auxiliary Loss, Contrastive Loss].
    """
    def __init__(self, num_tasks=4, hidden_dim=128):
        """
        Args:
            num_tasks (int): Number of concurrent objectives (default: 4).
            hidden_dim (int): Dimensionality of the latent loss-feature space.
        """
        super(IMNet, self).__init__()
        self.num_tasks = num_tasks

        # 【增强】更深的网络结构，更强的正则化，适配4任务
        self.meta_layer = nn.Sequential(
            nn.Linear(num_tasks, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

        # 谱正则化系数：稍微降低初始值，避免早期过度正则化
        self.beta = nn.Parameter(torch.tensor(0.005))

        # 温度参数：降低初始温度，使权重分配更灵敏
        self.temp = nn.Parameter(torch.ones(1) * 0.6)

    def forward(self, x, min_priority=0.05):
        """
        Generates task weights based on the local curvature of the objective manifold.

        Args:
            x (Tensor): Input losses/features of shape (num_tasks,)
                       representing [BPR_loss, Aux_loss, CL_loss].
            min_priority (float): Minimum weight bound to prevent task starvation.

        Returns:
            raw_weights (Tensor): Unconstrained softmax probabilities (for observation).
            constrained_weights (Tensor): Final task weights after residual protection.
        """
        # 1. Scale-invariant Transformation:
        x_scaled = torch.log(x.detach() + 1e-8)

        # 2. Latent Projection:
        logits = self.meta_layer(x_scaled)

        # 3. Temperature-Calibrated Softmax:
        temp = torch.clamp(self.temp, min=0.3, max=3.0)
        raw_weights = F.softmax(logits / temp, dim=-1)

        # 4. Proactive Constraint / Residual Protection Mechanism:
        num_tasks = x.size(-1)
        min_w = min_priority

        # Linear rescaling: Allocates minimum weights while maintaining the total sum of 1.
        constrained_weights = min_w + (1.0 - min_w * num_tasks) * raw_weights

        return raw_weights, constrained_weights

    def compute_weighted_loss(self, losses, weights):
        """
        The Weighted Energy Functional (Equation 5 in Manuscript).
        Combines discrete task losses into a unified scalar objective for backpropagation.
        """
        if isinstance(losses, list):
            losses = torch.stack(losses)
        return torch.sum(losses * weights)

    def compute_meta_loss(self, val_loss, spectral_radius):
        """
        Core Mechanism: Proactive Spectral Regularization.

        Objective: L_meta = Generalization_Loss + beta * Hessian_Spectral_Radius

        By penalizing the maximum eigenvalue (lambda_max) of the Hessian, IM-Net
        converges towards 'flat' and 'stable' regions of the parameter space,
        enhancing generalization.
        """
        # 频谱正则化项：通过惩罚 Hessian 谱半径来抑制优化过程中的“干扰能量”
        # Spectral Regularization term: Suppress 'interference energy' during optimization
        # by penalizing the Hessian spectral radius.
        interference_energy = self.beta * torch.tensor(spectral_radius, device=val_loss.device)
        return val_loss + interference_energy

    def get_control_parameters(self):
        """
        Extracts current meta-control coefficients for logging and monitoring.
        Provides insights into the 'Phase Transition' of task weights.
        """
        self.eval()
        device = next(self.parameters()).device
        # 模拟平衡状态下的输入，观察网络的初始权重偏好
        # Simulate balanced input to observe the network's initial weight preference.
        dummy_input = torch.tensor([1.0, 1.0, 1.0]).to(device)
        with torch.no_grad():
            raw, constrained = self.forward(dummy_input)
        self.train()
        return raw, constrained
