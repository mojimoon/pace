import mnist_cifar_imagenet_svhn.selection as s
import selection_metrics as sm
import os
import numpy as np
import tensorflow as tf

model_names = ['lenet1', 'lenet4', 'lenet5']
testX, testy = s.get_data('lenet1') # get_mnist

# print(testX.shape) # (10000, 28, 28, 1)
# models = [s.get_model(_) for _ in model_names]

out_csv = 'report/mnist.csv'
test_dir = 'test/mnist' # test/mnist/{test_set}/{model_name}/{selection_metric}/{budget}

metrics = ['rnd', 'ent', 'gini', 'dat', 'gd', 'kmnc', 'nac', 'lsa', 'dsa', 'nc', 'std', 'pace', 'dr', 'ces', 'mcp', 'est']
budgets = [50, 100, 150, 200]
# budgets = [0.05, 0.1, 0.15, 0.2]

def run_selection(model_name, test_set, testX, testy, metrics, budgets):
    model = s.get_model(model_name)

    # def select(sess, candidateX, candidatey, model, budget, selection_metric, **kwargs)
    for m in metrics:
        for b in budgets:
            try:
                sess = tf.compat.v1.Session()
                selectedX, selectedy = sm.select(
                    sess, testX, testy, model, b, m
                )
                score = model.evaluate(selectedX, selectedy, verbose=0)
                test_out_dir = os.path.join(test_dir, test_set, model_name, m, str(b))
                if not os.path.exists(test_out_dir):
                    os.makedirs(test_out_dir)
                np.save(os.path.join(test_out_dir, 'X.npy'), selectedX)
                np.save(os.path.join(test_out_dir, 'y.npy'), selectedy)
                with open(out_csv, 'a') as f:
                    f.write(f'{model_name},{test_set},{m},{b},{score[0]},{score[1]}\n')
            except Exception as e:
                with open('log/mnist.log', 'a') as f:
                    f.write(f'Error with model {model_name}, test_set {test_set}, metric {m}, budget {b}: {str(e)}\n')

def main():
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as f:
            f.write('model,test_set,selection_metric,budget,loss,accuracy\n')

    run_selection('lenet1', 'mnist', testX, testy, metrics, budgets)

if __name__ == '__main__':
    main()
