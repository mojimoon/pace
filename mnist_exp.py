import mnist_cifar_imagenet_svhn.selection as mnist
import metrics
import os
import numpy as np
import tensorflow as tf

model_names = ['lenet1', 'lenet4', 'lenet5']
testX, testy = mnist.get_data('lenet1') # get_mnist

# print(testX.shape) # (10000, 28, 28, 1)

out_csv = 'report/mnist.csv'
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

def run_selection(model_name, test_set, testX, testy, metricList, budgets):
    model = mnist.get_model(model_name)

    for m in metricList:
        for b in budgets:
            # try:
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
            #     with open('log/mnist.log', 'a') as f:
            #         f.write(f'Error with model {model_name}, test_set {test_set}, metric {m}, budget {b}: {str(e)}\n')

def main():
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write('model,test_set,selection_metric,budget,accuracy\n')

    run_selection('lenet1', 'mnist', testX, testy, ['gd', 'std'], budgets)

if __name__ == '__main__':
    main()
