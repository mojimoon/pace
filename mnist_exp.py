import mnist_cifar_imagenet_svhn.selection as mnist
import metrics
import os
import numpy as np
import pandas as pd
# import tensorflow as tf

model_names = ['lenet1', 'lenet4', 'lenet5']
testX, testy = mnist.get_data('lenet1') # get_mnist
trainX, trainy = mnist.get_mnist_train()

# print(testX.shape) # (10000, 28, 28, 1)

out_csv = 'report/mnist2.csv'
test_dir = 'test/mnist' # test/mnist/{test_set}/{model_name}/{selection_metric}/{budget}

metricList = ['rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'nc', 'std', 'pace', 'dr', 'ces', 'mcp', 'est']
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

# def describe_model(model_name):
#     model = mnist.get_model(model_name)
#     model.summary()
#     return model

def run_selection(model_name, test_set, testX, testy, metricList, budgets):
    model = mnist.get_model(model_name)

    for m in metricList:
        for b in budgets:
            # try:
            if True:
                if m == 'dat':
                    n_test = testX.shape[0]
                    hybridX = np.concatenate((trainX[:int(n_test // 2)], testX[:int(n_test // 2)]), axis=0)
                    hybridy = np.concatenate((trainy[:int(n_test // 2)], testy[:int(n_test // 2)]), axis=0)
                    selectedX, selectedy, idx = metrics.dat_ood_detector(
                        testX, testy, model, b, trainX, trainy, hybridX, hybridy, batch_size=128, num_classes=10
                    )
                else:
                    selectedX, selectedy, idx = metrics.select(
                        testX, testy, model, b, m
                    )
                score = model.evaluate(selectedX, selectedy, verbose=0)
                test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
                if not os.path.exists(test_out_dir):
                    os.makedirs(test_out_dir)
                # save idx to X.txt and true values to y.txt
                np.savetxt(os.path.join(test_out_dir, 'X.txt'), idx, fmt='%d')
                np.savetxt(os.path.join(test_out_dir, 'y.txt'), onehot_to_int(selectedy).astype(int), fmt='%d')
                with open(out_csv, 'a') as f:
                    f.write(f'{model_name},{test_set},{m},{b},{score[1]}\n')
            # except Exception as e:
            #     with open('log/mnist2.log', 'a') as f:
            #         f.write(f'Error with model {model_name}, test_set {test_set}, metric {m}, budget {b}: {str(e)}\n')

def main():
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write('model,test_set,selection_metric,budget,accuracy\n')
    
    metricList = ['nac', 'std']

    for m in model_names:
        run_selection(m, 'mnist', testX, testy, metricList, budgets)
    
    _X, _y = mnist.get_corrupted_mnist()
    run_selection('lenet5', 'mnist_c', _X, _y, metricList, budgets)

    _X, _y = mnist.get_adv_mnist()
    run_selection('lenet5', 'mnist_adv', _X, _y, metricList, budgets)

    _X, _y = mnist.get_label_mnist()
    run_selection('lenet5', 'mnist_label', _X, _y, metricList, budgets)

    _X, _y = mnist.get_mnist_emnist()
    run_selection('lenet5', 'mnist_emnist', _X, _y, metricList, budgets)

def apfd(right, sort):
    length = np.sum(sort != 0)
    if length != len(sort):
        sort[sort == 0] = np.random.permutation(len(sort) - length) + length + 1
    sum_all = np.sum(sort[right != 1])
    n = len(sort)
    m = np.sum(right == 0)
    return 1 - float(sum_all) / (n * m) + 1. / (2 * n)

def apfd_from_order(is_fault, index_order):
    assert is_fault.ndim == 1, "at the moment, only unique faults are supported"
    ordered_faults = is_fault[index_order]
    fault_indexes = np.where(ordered_faults == 1)[0]
    k = np.count_nonzero(is_fault)
    n = is_fault.shape[0]
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
    right = (full_pred_int == full_y_int).astype(int)
    is_fault = (full_pred_int != full_y_int).astype(int)

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

                apfd_score = apfd(right, sort)
                apfd_from_order_score = apfd_from_order(is_fault, X_id)
                acc_hat = np.mean(full_pred_int[X_id] == full_y_int[X_id])
                acc = np.mean(full_pred_int == full_y_int)
                rmse_score = np.abs(acc_hat - acc)

                # Type 2 retraining
                concatenatedX = np.concatenate((originalX, fullX[X_id]), axis=0)
                concatenatedy = np.concatenate((originaly, fully[X_id]), axis=0)
                model.fit(concatenatedX, concatenatedy, epochs=3, batch_size=128, verbose=0)
                retrain_pred = model.predict(fullX, verbose=0)
                retrain_pred_int = np.argmax(retrain_pred, axis=1)
                retrain_acc = np.mean(retrain_pred_int == full_y_int)
                acc_improvement = retrain_acc - acc

                results.append({
                    'model': model_name,
                    'test_set': test_set,
                    'selection_metric': m,
                    'budget': b,
                    'apfd': apfd_score,
                    'apfd_from_order': apfd_from_order_score,
                    'acc_hat': acc_hat,
                    'acc': acc,
                    'rmse': rmse_score,
                    'retrain_acc': retrain_acc,
                    'acc_improvement': acc_improvement
                })
            # except Exception as e:
            #     with open('log/mnist2_eval.log', 'a') as f:
            #         f.write(f'Error with model {model_name}, test_set {test_set}, metric {m}, budget {b}: {str(e)}\n')
    
    return results

def evaluate():
    eval_csv = 'report/mnist_eval.csv'
    vals = []
    originalX, originaly = testX, testy
    metricList = ['nac', 'std']

    for m in model_names:
        vals.extend(run_evaluation(m, 'mnist', metricList, budgets, testX, testy, originalX, originaly))
    
    _X, _y = mnist.get_corrupted_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_c', metricList, budgets, _X, _y, originalX, originaly))
    _X, _y = mnist.get_adv_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_adv', metricList, budgets, _X, _y, originalX, originaly))
    _X, _y = mnist.get_label_mnist()
    vals.extend(run_evaluation('lenet5', 'mnist_label', metricList, budgets, _X, _y, originalX, originaly))
    _X, _y = mnist.get_mnist_emnist()
    vals.extend(run_evaluation('lenet5', 'mnist_emnist', metricList, budgets, _X, _y, originalX, originaly))

    df = pd.DataFrame(vals)
    if not os.path.exists(eval_csv):
        df.to_csv(eval_csv, index=False)
    else:
        df.to_csv(eval_csv, mode='a', header=False, index=False)
    
if __name__ == '__main__':
    main()
    # evaluate()
