import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # 核心：用户和物品的初始 Embedding
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def get_all_embeddings(self, graph, perturbed=False, eps=0.1):
        """
         perturbed=True: 引入对比学习的数据增强 (SimGCL 顶会主流做法)
         在每一层传播后，向 Embedding 注入 L2 范数归一化的均匀噪声。
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)

            # 【核心创新】：图结构级别的扰动 (Data Augmentation for CL)
            if perturbed and self.training:
                random_noise = torch.rand_like(all_emb).to(all_emb.device) * 2 - 1
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * eps

            embs.append(all_emb)

        # 采用平均池化 (Mean Pooling) 融合多层感受野
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users, pos_items, neg_items, graph):
        # 兼容旧版本的主任务前向传播
        all_users, all_items = self.get_all_embeddings(graph)
        return all_users[users], all_items[pos_items], all_items[neg_items]

class NCF(nn.Module):
    """ 标准神经网络协同过滤 (NCF)，支持可配置的 MLP 层数 """

    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3, mlp_hidden_ratio=2):
        """
        Args:
            num_users: 用户数
            num_items: 物品数
            embed_dim: 嵌入维度
            n_layers: MLP 的隐藏层数量（不包括输入层和输出层）
            mlp_hidden_ratio: 每层隐藏单元数相对于输入维度的缩减比例（逐层减半）
        """
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # 构建动态 MLP
        input_dim = embed_dim * 2
        layers = []
        hidden_dim = input_dim // mlp_hidden_ratio
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim * 2, hidden_dim))
            layers.append(nn.ReLU())
            # 逐层减半（可选）
            input_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 16)  # 不低于16
        # 最后一层输出1维评分
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_all_embeddings(self, graph=None):
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self, users, pos_items, neg_items=None, return_embs=False, graph=None):
        u_emb = self.user_embedding(users)
        pos_i_emb = self.item_embedding(pos_items)
        pos_concat = torch.cat([u_emb, pos_i_emb], dim=-1)
        pos_scores = self.mlp(pos_concat).squeeze(-1)

        neg_scores = None
        neg_i_emb = None
        if neg_items is not None:
            neg_i_emb = self.item_embedding(neg_items)
            neg_concat = torch.cat([u_emb, neg_i_emb], dim=-1)
            neg_scores = self.mlp(neg_concat).squeeze(-1)

        if return_embs:
            return pos_scores, neg_scores, u_emb, pos_i_emb, neg_i_emb
        else:
            return pos_scores, neg_scores


class SimGCL(nn.Module):
    """ Simple Graph Contrastive Learning (SimGCL). """

    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3, eps=0.1):
        super(SimGCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.eps = eps
        self.embedding = nn.Embedding(num_users + num_items, embed_dim)

        # 【修改点 1：初始化方式】
        # ⚠️ 致命点：原代码使用 Xavier 初始化，这会导致初始 Embedding 方差过大。
        # 在对比学习中，初始向量过长会导致 InfoNCE Loss 开局直接爆炸（你看到的 Loss > 1.35）。
        # ✅ 修正：LightGCN/SimGCL 官方体系均推荐使用极小方差的正态分布。
        nn.init.normal_(self.embedding.weight, std=0.1)

    def get_all_embeddings(self, graph, perturbed=False):
        all_emb = self.embedding.weight
        embs = [all_emb]  # 第 0 层：原始 Embedding

        for _ in range(self.n_layers):
            # 【修改点 2：卷积与噪声的先后顺序】
            # ⚠️ 致命点：原代码是“先加噪声，再做图卷积 (sparse.mm)”。
            # 这会导致你注入的噪声被图聚合操作“平滑”掉，失去了对比学习“制造差异化视图”的本意！
            # ✅ 修正：必须先进行邻居聚合，然后再对聚合后的表达注入噪声。
            all_emb = torch.sparse.mm(graph, all_emb)

            if perturbed and self.training:
                # 【修改点 3：噪声分布】
                # ⚠️ 致命点：原代码使用 torch.randn_like (标准正态分布)。
                # ✅ 修正：SimGCL 论文严格要求使用 torch.rand_like (U(0,1) 均匀分布)。
                # 因为配合后面的 torch.sign()，均匀分布能确保扰动方向严格在当前向量所在的象限内，
                # 而正态分布有负数，会打乱符号逻辑，导致语义被彻底破坏（这就是 Recall 跌到 0.004 的元凶）。
                noise = torch.rand_like(all_emb).to(all_emb.device)

                # 核心：等变噪声注入
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(noise, dim=-1) * self.eps

            embs.append(all_emb)

        # 最终表达：所有层的平均
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(final_emb, [self.num_users, self.num_items])
        return user_emb, item_emb

    def forward(self, users, pos_items, neg_items, graph):
        # 训练主任务 (BPR Loss) 时，不加噪声
        u_g, i_g = self.get_all_embeddings(graph, perturbed=False)
        return u_g[users], i_g[pos_items], i_g[neg_items]

class HINE(nn.Module):
    """
    Heterogeneous Information Network Embedding (HINE).
    简化实现：在 LightGCN 基础上引入层级注意力机制，模拟异构传播。
    """
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3):
        super(HINE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # 层级权重：模拟不同跳数（即异构路径长度）的重要性
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1))

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_all_embeddings(self, graph):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(graph, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # 核心修改：使用可学习的层权重进行加权聚合，而非简单均值
        all_embeddings = torch.stack(all_embeddings, dim=1)
        weights = F.softmax(self.layer_weights, dim=0)
        final_embeddings = torch.sum(all_embeddings * weights.unsqueeze(0).unsqueeze(-1), dim=1)

        u_g, i_g = torch.split(final_embeddings, [self.num_users, self.num_items])
        return u_g, i_g

    def forward(self, users, pos_items, neg_items, graph):
        u_g, i_g = self.get_all_embeddings(graph)
        return u_g[users], i_g[pos_items], i_g[neg_items]
