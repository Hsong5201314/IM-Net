import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from tqdm import tqdm


# ================= 1. 严格且高速的负采样 Dataset =================
class RecDataset(Dataset):
    def __init__(self, interaction_data, train_dict, num_items, neg_count=1):
        self.data = interaction_data
        self.train_dict = train_dict
        self.num_items = num_items
        self.neg_count = neg_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, pos_item = self.data[idx]
        pos_set = self.train_dict[user]
        neg_items = []
        max_tries = self.neg_count * 100
        tries = 0
        while len(neg_items) < self.neg_count and tries < max_tries:
            neg = random.randint(0, self.num_items - 1)
            if neg not in pos_set:
                neg_items.append(neg)
            tries += 1
        # 填充不足部分（随机，可能重复但概率低）
        while len(neg_items) < self.neg_count:
            neg_items.append(random.randint(0, self.num_items - 1))
        return user, pos_item, neg_items

def collate_fn(batch):
    users, pos_items, neg_items_list = zip(*batch)
    # 展开每个样本的多个负样本
    expanded_users = []
    expanded_pos = []
    expanded_neg = []
    for u, pos, negs in zip(users, pos_items, neg_items_list):
        for neg in negs:
            expanded_users.append(u)
            expanded_pos.append(pos)
            expanded_neg.append(neg)
    return torch.LongTensor(expanded_users), torch.LongTensor(expanded_pos), torch.LongTensor(expanded_neg)

# ================= 2. 核心数据处理器 =================
class DataProcessor:
    def __init__(self, path, dataset_type='yelp', batch_size=2048):
        """
        支持字典配置（包含 simulation_conflict 标志）或传统字符串路径。
        当 simulation_conflict=True 时，训练引擎会构造相反的辅助损失（ρ≈-1）。
        """
        # 判断 path 是否为字典
        if isinstance(path, dict):
            config = path
            self.path = config.get('data_path', './data/yelp')
            self.dataset_type = config.get('dataset', 'yelp')
            self.batch_size = config.get('batch_size', 2048)
            self.dual_aux = config.get('dual_aux', False)
            # 关键：读取仿真冲突标志
            self.simulation_conflict = config.get('simulation_conflict', False)
            if self.simulation_conflict:
                print("[SIMULATION MODE] DataProcessor: simulation_conflict=True. "
                      "Auxiliary loss will be set opposite to main BPR loss (conflict enforced).")
        else:
            self.path = path
            self.dataset_type = dataset_type
            self.batch_size = batch_size
            self.dual_aux = False
            self.simulation_conflict = False

        print(f"[INFO] Loading {self.dataset_type.upper()} dataset from {self.path}...")

        # 1. 预先加载所有文件
        self.train_data = self._load_interactions(os.path.join(self.path, "train.txt"))
        test_data = self._load_interactions(os.path.join(self.path, "test.txt"))

        meta_val_file = os.path.join(self.path, "meta_val.txt")
        if os.path.exists(meta_val_file):
            self.meta_val_data = self._load_interactions(meta_val_file)
            print(f"[INFO] Loaded {len(self.meta_val_data)} interactions for Meta-Validation.")
        else:
            self.meta_val_data = []
            print(f"[WARNING] {meta_val_file} not found. Fallback to train data for Meta-Validation.")

        # 全局计算最大 ID，防止越界
        all_data = self.train_data + test_data + self.meta_val_data
        self.n_users, self.n_items = self._get_counts(all_data)
        print(f"[INFO] Global Graph Size - Users: {self.n_users}, Items: {self.n_items}")

        # 2. 构建字典
        self.train_dict = self._build_user_item_dict(self.train_data, as_set=True)
        self.test_dict = self._build_user_item_dict(test_data, as_set=False)

        # 动态设置训练负采样数量
        if self.dataset_type == 'amazon':
            self.train_neg_count = 20
        else:
            self.train_neg_count = 10

        # 3. 创建 DataLoaders
        self.train_loader = DataLoader(
            RecDataset(self.train_data, self.train_dict, self.n_items, neg_count=self.train_neg_count),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            collate_fn=collate_fn
        )

        if len(self.meta_val_data) > 0:
            self.meta_val_loader = DataLoader(
                RecDataset(self.meta_val_data, self.train_dict, self.n_items, neg_count=1),
                batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                collate_fn=collate_fn
            )
        else:
            self.meta_val_loader = None

        # 4. 构建 LightGCN 的归一化邻接矩阵
        self.norm_adj = self._build_norm_adj()

        # 5. 构建辅助任务的采样池
        self._prepare_auxiliary_sampler()

        # 根据 dual_aux 标志生成二级辅助边
        if self.dual_aux:
            self._prepare_user_auxiliary()
            self._prepare_item_auxiliary()
        else:
            self.user_user_links = np.empty((0, 2), dtype=int)
            self.item_item_links = np.empty((0, 2), dtype=int)

    # ========== 辅助采样器（保持不变） ==========
    def _prepare_auxiliary_sampler(self):
        """为 Yelp 准备社交边，为 Amazon 准备商品共现关联"""
        if self.dataset_type == 'yelp':
            social_file = os.path.join(self.path, "social.txt")
            if os.path.exists(social_file):
                social_links = []
                try:
                    with open(social_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) > 1:
                                u1 = int(parts[0])
                                for u2_str in parts[1:]:
                                    social_links.append([u1, int(u2_str)])
                    raw_links = np.array(social_links, dtype=int)
                    valid_mask = (raw_links[:, 0] < self.n_users) & (raw_links[:, 1] < self.n_users)
                    self.aux_links = raw_links[valid_mask]
                    print(f"[INFO] Yelp Auxiliary Task: Loaded {len(self.aux_links)} valid User-User social links.")
                except Exception as e:
                    print(f"[WARNING] Social file load failed: {e}. Using empty auxiliary links.")
                    self.aux_links = np.empty((0, 2), dtype=int)
            else:
                print("[WARNING] Yelp social.txt not found.")
                self.aux_links = np.empty((0, 2), dtype=int)
        else:
            # Amazon
            self.users_with_multi_interact = [
                u for u, items in self.train_dict.items() if len(items) >= 2
            ]
            self.aux_links = np.empty((0, 2), dtype=int)
            print(f"[INFO] Amazon Auxiliary Task: Found {len(self.users_with_multi_interact)} users for Item Co-occurrence sampling.")

    def _prepare_user_auxiliary(self):
        """构建用户-用户共现边（基于共同购买的物品数量）"""
        print("[INFO] Building User-User co-occurrence edges (efficient inverted index)...")
        threshold = 3 if self.dataset_type == 'amazon' else 2
        item_to_users = defaultdict(list)
        for user, items in self.train_dict.items():
            for item in items:
                item_to_users[item].append(user)
        user_pair_count = defaultdict(int)
        for users in item_to_users.values():
            if len(users) < 2:
                continue
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u1, u2 = users[i], users[j]
                    if u1 > u2:
                        u1, u2 = u2, u1
                    user_pair_count[(u1, u2)] += 1
        user_links = [[u1, u2] for (u1, u2), cnt in user_pair_count.items() if cnt >= threshold]
        self.user_user_links = np.array(user_links) if user_links else np.empty((0, 2), dtype=int)
        print(f"[INFO] Generated {len(self.user_user_links)} User-User co-occurrence edges.")

    def _prepare_item_auxiliary(self):
        """构建物品-物品共现边（基于共同购买的用户数量）"""
        print("[INFO] Building Item-Item co-occurrence edges...")
        item_to_users = defaultdict(set)
        for user, items in self.train_dict.items():
            for item in items:
                item_to_users[item].add(user)
        item_list = list(item_to_users.keys())
        item_links = []
        threshold = 3 if self.dataset_type == 'amazon' else 2
        for i in range(len(item_list)):
            i1 = item_list[i]
            users1 = item_to_users[i1]
            for j in range(i + 1, len(item_list)):
                i2 = item_list[j]
                users2 = item_to_users[i2]
                common = len(users1 & users2)
                if common >= threshold:
                    item_links.append([i1, i2])
        self.item_item_links = np.array(item_links) if item_links else np.empty((0, 2), dtype=int)
        print(f"[INFO] Generated {len(self.item_item_links)} Item-Item co-occurrence edges.")

    def sample_aux_links(self, batch_size):
        """动态采样辅助任务的边"""
        if self.dataset_type == 'amazon':
            if len(self.users_with_multi_interact) == 0:
                return torch.randint(0, self.n_items, (batch_size,)), torch.randint(0, self.n_items, (batch_size,))
            sampled_users = np.random.choice(self.users_with_multi_interact, batch_size, replace=True)
            node1_list, node2_list = [], []
            for u in sampled_users:
                items = list(self.train_dict[u])
                i1, i2 = np.random.choice(items, 2, replace=False)
                node1_list.append(i1)
                node2_list.append(i2)
            return torch.LongTensor(node1_list), torch.LongTensor(node2_list)
        else:
            if len(self.aux_links) > 0:
                indices = np.random.randint(0, len(self.aux_links), size=batch_size)
                sampled_links = self.aux_links[indices]
                return torch.LongTensor(sampled_links[:, 0]), torch.LongTensor(sampled_links[:, 1])
            else:
                return torch.randint(0, self.n_users, (batch_size,)), torch.randint(0, self.n_users, (batch_size,))

    def sample_user_aux_links(self, batch_size):
        if len(self.user_user_links) == 0:
            return torch.randint(0, self.n_users, (batch_size,)), torch.randint(0, self.n_users, (batch_size,))
        indices = np.random.randint(0, len(self.user_user_links), size=batch_size)
        edges = self.user_user_links[indices]
        return torch.LongTensor(edges[:, 0]), torch.LongTensor(edges[:, 1])

    def sample_item_aux_links(self, batch_size):
        if len(self.item_item_links) == 0:
            return torch.randint(0, self.n_items, (batch_size,)), torch.randint(0, self.n_items, (batch_size,))
        indices = np.random.randint(0, len(self.item_item_links), size=batch_size)
        edges = self.item_item_links[indices]
        return torch.LongTensor(edges[:, 0]), torch.LongTensor(edges[:, 1])

    # ================= 基础工具函数 =================
    def _load_interactions(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    for item in parts[1:]:
                        data.append([user, int(item)])
        return data

    def _build_user_item_dict(self, data, as_set=False):
        ui_dict = defaultdict(set if as_set else list)
        for u, i in data:
            if as_set:
                ui_dict[u].add(i)
            else:
                ui_dict[u].append(i)
        return ui_dict

    def _get_counts(self, data):
        users = [d[0] for d in data]
        items = [d[1] for d in data]
        return max(users) + 1, max(items) + 1

    def _build_norm_adj(self):
        """构建 LightGCN 的归一化邻接矩阵"""
        print("[INFO] Building normalized adjacency matrix for Graph Models...")
        n_nodes = self.n_users + self.n_items
        users = [d[0] for d in self.train_data]
        items = [d[1] for d in self.train_data]
        row = np.array(users + [i + self.n_users for i in items])
        col = np.array([i + self.n_users for i in items] + users)
        data = np.ones(len(row), dtype=np.float32)
        adj_mat = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum)
        valid_idx = rowsum > 0
        d_inv[valid_idx] = np.power(rowsum[valid_idx], -0.5)
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        print("[INFO] Graph construction done!")
        return torch.sparse_coo_tensor(indices, values, shape)