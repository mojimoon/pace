'''
rewrite of selection_metrics.py in flavor of Keras
'''

import tensorflow as tf
import numpy as np

def make_batch(X, Y, batch_size):
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield X[start:end], Y[start:end]

def random_select(X, y, budget):
    idx = np.random.choice(len(X), budget, replace=False)
    return X[idx], y[idx], idx

def entropy(proba):
    proba = np.clip(proba, 1e-8, 1.0)
    entropy_val = -np.sum(proba * np.log(proba), axis=1)
    return entropy_val

def entropy_select(X, y, model, budget, batch_size=128):
    ent = []
    n_samples = X.shape[0]
    for x_batch, y_batch in make_batch(X, y, batch_size):
        proba = model.predict(x_batch)
        ent.append(entropy(proba))
    scores = np.concatenate(ent)
    idx = np.argsort(scores)[::-1] # Largest entropy first
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], idx

def gini(proba):
    gini_val = 1 - np.sum(proba ** 2, axis=1)
    return gini_val

def gini_select(X, y, model, budget, batch_size=128):
    raw = []
    n_samples = X.shape[0]
    for x_batch, y_batch in make_batch(X, y, batch_size):
        proba = model.predict(x_batch)
        raw.append(gini(proba))
    scores = np.concatenate(raw)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], idx

def select(X, y, model, budget, metric, batch_size=128, **kwargs):
    if metric == 'rnd':
        return random_select(X, y, budget)
    elif metric == 'ent':
        return entropy_select(X, y, model, budget, batch_size)
    elif metric == 'gini':
        return gini_select(X, y, model, budget, batch_size)
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")