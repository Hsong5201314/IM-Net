# rebuild_dataset.py
import os
import random
from collections import defaultdict

INPUT_DIR = "./yelp_processed_for_meta"   # 原始完整数据集
OUTPUT_DIR = "./yelp_rebalanced"          # 新数据集
random.seed(2026)

def load_interactions(file_path):
    user_items = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            user_items[u].extend(items)
    return user_items

def split_user_items(user_items, test_per_user=1):
    train, test = [], []
    for u, items in user_items.items():
        if len(items) <= test_per_user:
            train.extend([(u, i) for i in items])
            continue
        test_items = items[-test_per_user:]
        train_items = items[:-test_per_user]
        train.extend([(u, i) for i in train_items])
        test.extend([(u, i) for i in test_items])
    return train, test

def write_txt(data, filename):
    ud = defaultdict(list)
    for u, i in data:
        ud[u].append(i)
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        for u in sorted(ud.keys()):
            f.write(f"{u} " + " ".join(str(i) for i in ud[u]) + "\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original = load_interactions(os.path.join(INPUT_DIR, "train.txt"))
    train_data, test_data = split_user_items(original, test_per_user=1)
    random.shuffle(train_data)
    val_size = int(len(train_data) * 0.1)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    write_txt(train_data, "train.txt")
    write_txt(val_data, "val.txt")
    write_txt(test_data, "test.txt")
    meta_val = random.sample(val_data, min(10000, len(val_data)))
    write_txt(meta_val, "meta_val.txt")
    import shutil
    shutil.copy(os.path.join(INPUT_DIR, "social.txt"), os.path.join(OUTPUT_DIR, "social.txt"))
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    main()