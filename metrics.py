
import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix_manual(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def f1_score_macro(y_true, y_pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        scores.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0)
    return np.mean(scores)

import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix_manual(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def f1_score_macro(y_true, y_pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        scores.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0)
    return np.mean(scores)

import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def confusion_matrix_manual(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def f1_score_macro(y_true, y_pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        scores.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0)
    return np.mean(scores)
