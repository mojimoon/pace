import mnist_cifar_imagenet_svhn.selection as s
# import selection_metrics as sm

model_names = ['lenet1', 'lenet4', 'lenet5']
testX, testy = s.get_data('lenet1') # get_mnist

# print(testX.shape) # (10000, 28, 28, 1)

models = [s.get_model(_) for _ in model_names]

out_csv = 'report/mnist.csv'

with open(out_csv, 'w') as f:
    f.write('model,loss,accuracy\n')

for i in range(len(models)):
    model = models[i]
    model_name = model_names[i]
    results = s.get_score(testX, testy, model)
    with open(out_csv, 'a') as f:
        f.write(f'{model_name},{results[0]},{results[1]}\n')
