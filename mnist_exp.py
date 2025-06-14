import mnist_cifar_imagenet_svhn.selection as mnist
import metrics
import os
import numpy as np
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
    
    metricList = ['kmnc', 'nac', 'lsa', 'dsa']

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

if __name__ == '__main__':
    main()
