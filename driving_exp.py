import driving.driving_models as driving_models
import driving.selection as selection
import driving.epoch.epoch_model as epoch_model
import metrics
import os
import numpy as np
from sklearn.metrics import mean_squared_error

basedir = os.path.abspath(os.path.dirname(__file__))

out_csv = 'report/driving.csv'
test_dir = 'test/driving' # test/driving/{test_set}/{model_name}/{selection_metric}/{budget}

metricList = ['rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'nc', 'std', 'pace', 'dr', 'ces', 'mcp', 'est']
budgets = [50, 100, 150, 200, 5614]

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

def get_datasets(vs):
    datasets = dict()
    for v in vs:
        gen, tot = selection.get_data(v)
        x, y = from_generator(gen, tot)
        datasets[v] = (x, y, tot)
        print('loaded dataset', v)
    return datasets

# def test():
#     models = get_driving_models()
#     datasets = get_datasets()
#     pred = models['dave2v1'].predict(datasets['udacity'][0])
#     print('Udacity dataset shape:', datasets['udacity'][0].shape)
#     print('Prediction shape:', pred.shape)
#     # both pred and y are array of shape (N, 1) in range (-1, 1). They describe steering angles in autonomus driving in radian. What is the best metric to evaluate the performance?
#     y = datasets['udacity'][1]
#     mse = mean_squared_error(y, pred)
#     print('MSE', mse)

models = get_driving_models()
train_gen, tot = selection.get_train_data()
trainX, trainy = from_generator(train_gen, tot)

def run_selection(model_name, test_set, datasets, metricList, budgets):
    model = models[model_name]
    testX, testy, tot = datasets[test_set]

    for m in metricList:
        for b in budgets:
            try:
                if m == 'dat':
                    hybridX = np.concatenate((datasets['udacity'][0], testX), axis=0)
                    hybridy = np.concatenate((datasets['udacity'][1], testy), axis=0)
                    selectedX, selectedy, idx = metrics.dat_ood_detector(
                        testX, testy, model, b, datasets['udacity'][0], datasets['udacity'][1], hybridX, hybridy
                    )
                else:
                    print('metric select', m, 'budget', b, 'model', model_name)
                    selectedX, selectedy, idx = metrics.select(trainX, trainy,
                        testX, testy, model, b, m, 'udacity', model_name
                    )
                test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
                if not os.path.exists(test_out_dir):
                    os.makedirs(test_out_dir)
                np.savetxt(os.path.join(test_out_dir, 'X.txt'), idx, fmt='%d')
                np.savetxt(os.path.join(test_out_dir, 'y.txt'), selectedy, fmt='%f')
                # all models are compiled with loss='mse'
                score = model.evaluate(selectedX, selectedy, verbose=0)
                with open(out_csv, 'a') as f:
                    f.write(f'{model_name},{test_set},{m},{b},{score}\n')
            except Exception as e:
                with open('log/driving.log', 'a') as f:
                    f.write(f'Error with model {model_name}, test_set {test_set}, metric {m}, budget {b}: {str(e)}\n')

def main():
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write('model,test_set,selection_metric,budget,mse\n')

    metricList = ['dsa']
    datasets = get_datasets(['udacity'])
    for m in models.keys():
        run_selection(m, 'udacity', datasets, metricList, budgets)
    
    for v in ['udacity_C', 'udacity_label', 'udacity_adv', 'udacity_dave']:
        datasets = get_datasets([v])
        run_selection('epoch', v, datasets, metricList, budgets)

if __name__ == '__main__':
    # test()
    main()
