'''
Implement 4 evaluation criteria:
1. APFD;
2. Clustering-based Fault Estimation;
3. RMSE;
4. Accuracy Improvement;
'''
def APFD(right,sort):
    '''
    APFD measures misprediction-detection rate

    '''
    length = np.sum(sort != 0)
    if length != len(sort):
        sort[sort == 0] = np.random.permutation(len(sort) - length) + length + 1
    sum_all = np.sum(sort[[right != 1]])
    n = len(sort)
    m = pd.value_counts(right)[0]
    return 1 - float(sum_all) / (n * m) + 1. / (2 * n)

def RMSE():
    '''
    Goal: using a small selected set to estimate the accuracy of the who;e testing set.
    acc: the accuracy of the whole operational dataset
    acc_hat: the accuracy of selected test suite
    '''
    if randomness == True:
        N = len(acc)
        return np.sqrt(1/N * np.sum((acc_hat - acc)**2, axis=1))
    else:
        return np.abs(acc_hat - acc)