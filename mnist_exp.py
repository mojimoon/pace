
import mnist_cifar_imagenet_svhn.selection as mnist
import sys
sys.path.append('/home/jzhang2297/empirical/')
import metrics
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import sklearn
model_names = ['lenet1', 'lenet4', 'lenet5']
testX, testy = mnist.get_data('lenet1') # get_mnist
trainX, trainy = mnist.get_mnist_train()

# print(testX.shape) # (10000, 28, 28, 1)

out_csv = 'report/mnist.csv'
test_dir = 'test/mnist' # test/mnist/{test_set}/{model_name}/{selection_metric}/{budget}

budgets = [50, 100, 150, 200]

def onehot_to_int(y):
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    elif y.ndim == 1:
        return y
    else:
        raise ValueError("Input must be a one-hot encoded array or a single-dimensional array.")

def int_to_onehot(y, num_classes=10):
    if y.ndim == 1:
        return np.eye(num_classes)[y]
    elif y.ndim == 2 and y.shape[1] == num_classes:
        return y
    else:
        raise ValueError("Input must be a one-hot encoded array or a single-dimensional array.")

def run_selection(model_name, test_set, testX, testy, metricList, budgets, force_save=False):
    model = mnist.get_model(model_name)
    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            if not os.path.exists(test_out_dir):
                os.makedirs(test_out_dir)
            # save idx to X.txt and true values to y.txt
            print('save selected test suite to', test_out_dir)
            if os.path.exists(os.path.join(test_out_dir, 'X.txt')) and os.path.exists(os.path.join(test_out_dir, 'y.txt')) and force_save==False:
                print('path exists')
                continue
            # try:
            if True:
                if m == 'dat':
                    n_test = testX.shape[0]
                    hybridX = np.concatenate((trainX[:int(n_test // 2)], testX[:int(n_test // 2)]), axis=0)
                    hybridy = np.concatenate((trainy[:int(n_test // 2)], testy[:int(n_test // 2)]), axis=0)
                    selectedX, selectedy, idx = metrics.dat_ood_detector(
                        testX, testy, model, b, test_set, trainX, trainy, hybridX, hybridy, batch_size=128, num_classes=10
                    )
                else:
                    selectedX, selectedy, idx = metrics.select(trainX, trainy,
                        testX, testy, model, b, m, test_set, model_name
                    )
                score = model.evaluate(selectedX, selectedy, verbose=0)
                np.savetxt(os.path.join(test_out_dir, 'X.txt'), idx, fmt='%d')
                np.savetxt(os.path.join(test_out_dir, 'y.txt'), onehot_to_int(selectedy).astype(int), fmt='%d')
                with open(out_csv, 'a') as f:
                    f.write(f'{model_name},{test_set},{m},{b},{score[1]}\n')
            # except Exception as e:
            #     with open('log/mnist2.log', 'a') as f:
            #         f.write(f'Error with model {model_name}, test_set {test_set}, metric {m}, budget {b}: {str(e)}\n')

def select(resave):
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write('model,test_set,selection_metric,budget,accuracy\n')
    
    metricList = ['dat']

    for m in model_names:
        run_selection(m, 'mnist', testX, testy, metricList, budgets, resave)

    _X, _y = mnist.get_corrupted_mnist()
    run_selection('lenet5', 'mnist_c', _X, _y, metricList, budgets, resave)

    _X, _y = mnist.get_adv_mnist()
    run_selection('lenet5', 'mnist_adv', _X, _y, metricList, budgets, resave)

    _X, _y = mnist.get_label_mnist()
    run_selection('lenet5', 'mnist_label', _X, _y, metricList, budgets, resave)

    _X, _y = mnist.get_mnist_emnist()
    run_selection('lenet5', 'mnist_emnist', _X, _y, metricList, budgets, resave)


def apfd_from_order(is_fault, index_order):
    assert is_fault.ndim == 1, "at the moment, only unique faults are supported"
    ordered_faults = is_fault[index_order]
    fault_indexes = np.where(ordered_faults == 1)[0]
    k = np.count_nonzero(is_fault)
    n = is_fault.shape[0]
    # The +1 comes from the fact that the first sample has index 0 but order 1
    sum_of_fault_orders = np.sum(fault_indexes + 1)
    return 1 - (sum_of_fault_orders / (k * n)) + (1 / (2 * n))

def rmse(acc, acc_hat, randomness=True):
    if randomness:
        N = len(acc)
        return np.sqrt(1 / N * np.sum((acc_hat - acc) ** 2, axis=1))
    else:
        return np.abs(acc_hat - acc)

# improvement: full_pred and acc can be computed once for all metrics

def run_evaluation(model_name, test_set, metricList, budgets, fullX, fully, originalX, originaly):
    model = mnist.get_model(model_name)
    full_y_int = onehot_to_int(fully)  # (n_samples, 28, 28, 1)
    full_pred = model.predict(fullX, verbose=0)  # (n_samples, 10)
    full_pred_int = np.argmax(full_pred, axis=1)  # (n_samples,)
    #right = (full_pred_int == full_y_int).astype(int)
    #is_fault = (full_pred_int != full_y_int).astype(int)

    results = []

    for m in metricList:
        for b in budgets:
            # try:
            if True:
                test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
                if not os.path.exists(test_out_dir):
                    raise FileNotFoundError(f"Test output directory {test_out_dir} does not exist.")
                X_id = np.loadtxt(os.path.join(test_out_dir, 'X.txt'), dtype=int)  # (n_selected,)
                sort = np.zeros(fullX.shape[0], dtype=int)
                sort[X_id] = np.arange(1, b + 1)

                #apfd_from_order_score = apfd_from_order(is_fault, X_id)
                acc_hat = np.mean(full_pred_int[X_id] == full_y_int[X_id])
                failures = np.sum(full_pred_int[X_id] != full_y_int[X_id])

                acc = np.mean(full_pred_int == full_y_int)
                rmse_score = np.abs(acc_hat - acc)

                # Type 2 retraining
                concatenatedX = np.concatenate((originalX, fullX[X_id]), axis=0)
                concatenatedy = np.concatenate((originaly, fully[X_id]), axis=0)
                model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
                model.fit(concatenatedX, concatenatedy, epochs=5, batch_size=128, verbose=0)

                mask = np.ones(fullX.shape[0], dtype=bool)
                mask[X_id] = False
                retrain_pred = model.predict(fullX[mask], verbose=0)
                retrain_pred_int = np.argmax(retrain_pred, axis=1)
                retrain_acc = np.mean(retrain_pred_int == full_y_int[mask])
                acc_clean = np.mean(full_pred_int[mask] == full_y_int[mask])
                acc_improvement = retrain_acc - acc_clean
                re = {
                    'model': model_name,
                    'test_set': test_set,
                    'selection_metric': m,
                    'budget': b,
                    'acc': acc,
                    'acc_hat': acc_hat,
                    'failures': failures,
                    'rmse': rmse_score,
                    'retrain_acc': retrain_acc,
                    'acc_improvement': acc_improvement
                }
                results.append(re)
                print(re)
    
    return results

def save_results(part_name, vals):
    df = pd.DataFrame(vals)
    save_dir = './report/mnist/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, f'{part_name}.csv')
    df.to_csv(path, index=False)

def evaluate():
    vals = []
    originalX, originaly = trainX, trainy
    metricList = ['dat'] #['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        vals.extend(run_evaluation(m, 'mnist', metricList, budgets, testX, testy, originalX, originaly))
        save_results(f'{m}_mnist', vals)
    _X, _y = mnist.get_corrupted_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_c', metricList, budgets, _X, _y, originalX, originaly))
    save_results('lenet5_mnist_c', vals)
    _X, _y = mnist.get_adv_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_adv', metricList, budgets, _X, _y, originalX, originaly))
    save_results('lenet5_mnist_adv', vals)
    _X, _y = mnist.get_label_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_label', metricList, budgets, _X, _y, originalX, originaly))
    save_results('lenet5_mnist_label', vals)
    _X, _y = mnist.get_mnist_emnist()
    vals.extend(run_evaluation('lenet5', 'mnist_emnist', metricList, budgets, _X, _y, originalX, originaly))
    save_results('lenet5_mnist_emnist', vals)



def evaluate_cluster():
    vals = []
    metricList = ['dr', 'ces', 'mcp', 'est', 'rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'std', 'pace']

    for m in model_names:
        vals.extend(evaluation_cluster(m, 'mnist', metricList, budgets, testX, testy))
        save_results(f'{m}_mnist_cluster', vals)
    _X, _y = mnist.get_corrupted_mnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_c', metricList, budgets, _X, _y))
    save_results('lenet5_mnist_c_cluster', vals)
    _X, _y = mnist.get_adv_mnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_adv', metricList, budgets, _X, _y))
    save_results('lenet5_mnist_adv_cluster', vals)
    _X, _y = mnist.get_label_mnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_label', metricList, budgets, _X, _y))
    save_results('lenet5_mnist_label_cluster', vals)
    _X, _y = mnist.get_mnist_emnist()
    vals.extend(evaluation_cluster('lenet5', 'mnist_emnist', metricList, budgets, _X, _y))
    save_results('lenet5_mnist_emnist_cluster', vals)


def evaluation_cluster(model_name, test_set, metricList, budgets, fullX, fully):
    # env: malwaregpu
    from cuml.manifold import UMAP
    from DBCV.DBCV import DBCV
    import cuml
    from cuml import DBSCAN
    from torchvision import models
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    model_resnet50 = FeatureExtractor(resnet50)  # already included AdaptiveAvgPool2d
    #model = mnist.get_model(model_name)
    #full_y_int = onehot_to_int(fully)  # (n_samples, 28, 28, 1)
    #full_pred = model.predict(fullX, verbose=0)  # (n_samples, 10)
    #full_pred_int = np.argmax(full_pred, axis=1)  # (n_samples,)
    results = []

    for m in metricList:
        for b in budgets:
            test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
            if not os.path.exists(test_out_dir):
                raise FileNotFoundError(f"Test output directory {test_out_dir} does not exist.")
            #X_id = np.loadtxt(os.path.join(test_out_dir, 'X.txt'), dtype=int)  # (n_selected,)
            #idx = full_pred_int[X_id] != full_y_int[X_id]
            #mispred_indices = X_id[idx]
            mispred_indices = np.load(f'./test/mnist/cluster_idx/{test_set}_{model_name}_{m}_{b}.npy')
            if len(mispred_indices) <= 2:
                # print(f'Insufficient mispredictions for idr={idr}, dataset={dataset}, model={model_name}, bg={bg}')
                clustering_results = {"Number of Clusters": -1,
                                      "Silhouette Score": -1.0,
                                      "Combined Score": -1.0,
                                      "Number of Mispredicted Inputs": len(mispred_indices),
                                      "Number of Noisy Inputs": -1,
                                      "Config": -1,
                                      "test_set": test_set,
                                      "selection_metric": m,
                                      "budget": b,
                                      "model": model_name}
            else:
                images = fullX[mispred_indices]
                feature = extract_features_torch(images=images, model=model_resnet50, dataset_name=test_set)
                suite_feat = np.vstack(feature)

                u = umap_gpu(ip_mat=suite_feat, min_dist=0.1, n_components=min(25, len(suite_feat) - 1), n_neighbors=15,
                             metric='Euclidean')
                optimal_eps = find_optimal_eps(u)
                dbscan_float = DBSCAN(eps=optimal_eps,
                                      # we use different optimal eps across settings due to large variation.
                                      min_samples=2)
                labels = dbscan_float.fit_predict(u)
                if len(np.unique(labels)) == 1:
                    clustering_results = {
                        "Number of Clusters": labels.max() + 1,
                        "Silhouette Score": -1,
                        "DBCV Score": -1,
                        "Combined Score": -1,  # 0.5 * silhouette_umap + 0.5 * DBCV_score,
                        "Number of Mispredicted Inputs": len(u),
                        "Number of Noisy Inputs": list(labels).count(-1),
                        "test_set": test_set,
                        "selection_metric": m,
                        "budget": b,
                        "model": model_name
                    }
                else:
                    silhouette_umap = sklearn.metrics.silhouette_score(u, labels)
                    DBCV_score = DBCV(u, labels)
                    clustering_results = {
                        "Number of Clusters": labels.max() + 1,
                        "Silhouette Score": silhouette_umap,
                        "DBCV Score": DBCV_score,
                        "Combined Score": 0.5 * silhouette_umap + 0.5 * DBCV_score,
                        "Number of Mispredicted Inputs": len(u),
                        "Number of Noisy Inputs": list(labels).count(-1),
                        "test_set": test_set,
                        "selection_metric": m,
                        "budget": b,
                        "model": model_name
                    }
            print(clustering_results)
            results.append(clustering_results)

    return results
    
if __name__ == '__main__':
    # select(resave=True)
    # evaluate()
    evaluate_cluster()
