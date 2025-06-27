import driving.driving_models as driving_models
import driving.selection as selection
import driving.epoch.epoch_model as epoch_model
import metrics
import os
import numpy as np

basedir = os.path.abspath(os.path.dirname(__file__))

def get_driving_models():
    models = {
        'dave2v1': driving_models.Dave_orig(load_weights=True),
        'dave2v2': driving_models.Dave_norminit(load_weights=True),
        'dave2v3': driving_models.Dave_dropout(load_weights=True),
        'epoch': epoch_model.build_cnn(weights_path='/home/jzhang2297/empirical/pace/driving/epoch/epoch.h5')
    }
    return models

def from_generator(gen, tot):
    xs, ys = [], []
    collected = 0
    while collected < tot:
        x_batch, y_batch = next(gen)
        remain = tot - collected
        if len(x_batch) > remain:
            x_batch = x_batch[:remain]
            y_batch = y_batch[:remain]
            collected += remain
        else:
            collected += len(x_batch)
        xs.append(x_batch)
        ys.append(y_batch)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def get_datasets():
    datasets = dict()
    for v in ['udacity', 'udacity_C', 'udacity_label', 'udacity_adv', 'udacity_dave']:
        gen, tot = selection.get_data(v)
        x, y = from_generator(gen, tot)
        datasets[v] = (x, y, tot)
    return datasets

def test():
    models = get_driving_models()
    datasets = get_datasets()

if __name__ == '__main__':
    test()