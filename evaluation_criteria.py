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

from typing import List, Union
def apfd_from_order(is_fault, index_order: Union[List[int], np.ndarray]) -> float:
    """
    Compute APFD from the index order of the misclassified samples.
    """
    assert is_fault.ndim == 1, "at the moment, only unique faults are supported"
    ordered_faults = is_fault[index_order]
    fault_indexes = np.where(ordered_faults == 1)[0]
    k = np.count_nonzero(is_fault)
    n = is_fault.shape[0]
    # The +1 comes from the fact that the first sample has index 0 but order 1
    sum_of_fault_orders = np.sum(fault_indexes + 1)
    return 1 - (sum_of_fault_orders / (k * n)) + (1 / (2 * n))

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