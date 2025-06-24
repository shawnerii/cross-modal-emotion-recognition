
import numpy as np

def dataset_stats(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {str(k): {'count': int(v), 'pct': round(100*v/total, 2)} 
            for k, v in zip(unique, counts)}

def train_val_split(data, labels, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(data))
    split = int(len(data) * (1 - val_ratio))
    return data[idx[:split]], data[idx[split:]], labels[idx[:split]], labels[idx[split:]]
