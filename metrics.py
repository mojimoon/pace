'''
rewrite of selection_metrics.py in flavor of Keras
'''

from collections import defaultdict
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from keras.models import Model
# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Dense
import keras.layers as layers
from scipy.stats import gaussian_kde
from functools import reduce
import copy
import keras.backend as K

def to_ordinal(y):
    if y.ndim == 1:
        return y
    elif y.ndim == 2 and y.shape[1] == 1:
        return y.flatten()
    elif y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    else:
        return y

def to_onehot(y, num_classes=None):
    if y.ndim == 2 and y.shape[1] > 1:
        return y
    elif y.ndim == 1:
        if num_classes is None:
            num_classes = np.max(y) + 1
        return np.eye(num_classes)[y]
    elif y.ndim == 2 and y.shape[1] == 1:
        if num_classes is None:
            num_classes = np.max(y) + 1
        return np.eye(num_classes)[y.flatten()]
    else:
        return y

def make_batch(X, Y, batch_size):
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield X[start:end], Y[start:end]

def random_select(X, y, budget):
    idx = np.random.choice(len(X), budget, replace=False)
    return X[idx], y[idx], idx

def entropy(proba):
    proba = np.clip(proba, 1e-8, 1.0)
    entropy_val = -np.sum(proba * np.log(proba), axis=1)
    return entropy_val

def entropy_select(X, y, model, budget, batch_size=128):
    ent = []
    for x_batch, y_batch in make_batch(X, y, batch_size):
        proba = model.predict(x_batch)
        ent.append(entropy(proba))
    scores = np.concatenate(ent)
    idx = np.argsort(scores)[::-1] # Largest entropy first
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def gini(proba):
    gini_val = 1 - np.sum(proba ** 2, axis=1)
    return gini_val

def gini_select(X, y, model, budget, batch_size=128):
    raw = []
    for x_batch, y_batch in make_batch(X, y, batch_size):
        proba = model.predict(x_batch)
        raw.append(gini(proba))
    scores = np.concatenate(raw)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx, scores

def extract_layers(model):
    layers = []
    for l in model.layers:
        if isinstance(l, layers.Conv2D):
            layers.append(('conv', l.output))
        elif isinstance(l, layers.Dense):
            layers.append(('dense', l.output))
        # if 'conv' in l.name:
        #     layers.append(('conv', l.output))
        # elif 'dense' in l.name:
        #     layers.append(('dense', l.output))
    return layers

def dat_ood_detector(X, y, model, budget, trainX, trainy, hybridX, hybridy, batch_size=128, num_classes=None):
    dense1 = None
    # print(model.summary())
    for layer in model.layers:
        # if 'dense' in layer.name:
        if isinstance(layer, layers.Dense):
            dense1 = layer.output
            break
    if dense1 is None:
        raise ValueError("No dense layer found in the model.")
    feat_mat = []
    for x_batch, y_batch in make_batch(trainX, trainy, batch_size):
        feat = Model(inputs=model.input, outputs=dense1).predict(x_batch)
        feat_mat.append(feat)
    feat_mat = np.vstack(feat_mat)
    feat_mat_can = []
    for x_batch, y_batch in make_batch(X, y, batch_size):
        feat = Model(inputs=model.input, outputs=dense1).predict(x_batch)
        feat_mat_can.append(feat)
    feat_mat_can = np.vstack(feat_mat_can)

    center = np.mean(feat_mat, axis=0)
    dist = np.sum((feat_mat - center) ** 2, axis=1)
    dist_sort = np.sort(dist)
    threshold = dist_sort[int(0.95 * len(dist_sort))]
    dist_can = np.sum((feat_mat_can - center) ** 2, axis=1)
    canX_id, cany_id = X[dist_can <= threshold], y[dist_can <= threshold]
    canX_ood, cany_ood = X[dist_can > threshold], y[dist_can > threshold]

    budget_ratio = budget / len(X)
    id_select_num = int(len(canX_id) * budget_ratio)
    ood_select_num = budget - id_select_num
    para_there = 0.5
    tot_ood_size = len(canX_ood)
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size:
            ood_select_num = tot_ood_size
            id_select_num = int(budget - ood_select_num)
        else:
            id_select_num, ood_select_num = ood_select_num, id_select_num
    if id_select_num > int(para_there * budget):
        id_select_num = int(para_there * budget)
        ood_select_num = budget - id_select_num
        if ood_select_num > tot_ood_size:
            ood_select_num = tot_ood_size
            id_select_num = int(budget - ood_select_num)
    
    _, _, _, id_scores = gini_select(canX_id, cany_id, model, id_select_num, batch_size)
    idx = np.argsort(id_scores)[::-1]
    id_select_idx = idx[:id_select_num]
    selected_canX_id = canX_id[id_select_idx]
    selected_cany_id = cany_id[id_select_idx]

    candidate_prediction_label = model.predict(X, batch_size=batch_size)
    reference_prediction_label = model.predict(hybridX, batch_size=batch_size)
    reference_labels = []
    if num_classes is None: # y is one-hot
        num_classes = y.shape[1] if y.ndim > 1 else np.max(y) + 1
    for i in range(num_classes):
        label_num = len(np.where(reference_prediction_label == i)[0])
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(X) / budget
    reference_labels = reference_labels / s_ratio
    ood_part_index = np.where((dist_can > threshold) == True)[0]

    label_list = []
    index_list = []
    if ood_select_num == 0:
        selected_data = selected_canX_id
        selected_label = selected_cany_id
        selected_indices = id_select_idx
    else:
        num_ood = canX_ood.shape[0]
        ood_local_indices = np.arange(num_ood)

        for _ in range(1000):
            chosen_local_idx = np.random.choice(ood_local_indices, ood_select_num, replace=False)
            this_labels = candidate_prediction_label[ood_part_index[chosen_local_idx]]  # 用全局索引
            single_labels = []
            for i in range(num_classes):
                label_num = len(np.where(this_labels == i)[0])
                single_labels.append(label_num)
            index_list.append(chosen_local_idx)
            label_list.append(single_labels)
        index_list = np.asarray(index_list)
        label_list = np.asarray(label_list)

        label_minus = np.abs(label_list - reference_labels)
        var_list = np.sum(label_minus, axis=1)
        var_list_order = np.argsort(var_list)

        ood_select_index = index_list[var_list_order[0]]
        selected_canX_ood = canX_ood[ood_select_index]
        selected_cany_ood = cany_ood[ood_select_index]
        selected_data = np.concatenate((selected_canX_id, selected_canX_ood), axis=0)
        selected_label = np.concatenate((selected_cany_id, selected_cany_ood), axis=0)
        selected_indices = np.concatenate((id_select_idx, ood_part_index[ood_select_index]), axis=0)

    return selected_data, selected_label, selected_indices

class kmnc(object):
    def __init__(self,train,input,layers,k_bins=1000):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.k_bins=k_bins
        self.lst=[]
        self.upper=[]
        self.lower=[]
        index_lst=[]

        for index,l in layers:
            self.lst.append(Model(inputs=input,outputs=l))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(train).reshape(len(train),l.shape[-1])
            self.upper.append(np.max(temp,axis=0))
            self.lower.append(np.min(temp,axis=0))
        self.upper=np.concatenate(self.upper,axis=0)
        self.lower=np.concatenate(self.lower,axis=0)
        self.neuron_num=self.upper.shape[0]
        self.lst=list(zip(index_lst,self.lst))


    def fit(self,test):
        '''
        test:测试集数据
        输出测试集的覆盖率
        '''
        self.neuron_activate=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate=np.concatenate(self.neuron_activate,axis=1)
        act_num=0
        for index in range(len(self.upper)):
            bins=np.linspace(self.lower[index],self.upper[index],self.k_bins)
            act_num+=len(np.unique(np.digitize(self.neuron_activate[:,index],bins)))
        return act_num/float(self.k_bins*self.neuron_num)

    def rank_fast(self,test):
        '''
        test:测试集数据
        输出排序情况
        '''
        self.neuron_activate=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate=np.concatenate(self.neuron_activate,axis=1)
        big_bins=np.zeros((len(test),self.neuron_num,self.k_bins+1))
        for n_index,neuron_activate in enumerate(self.neuron_activate):
            for index in range(len(neuron_activate)):
                bins=np.linspace(self.lower[index],self.upper[index],self.k_bins)
                temp=np.digitize(neuron_activate[index],bins)
                big_bins[n_index][index][temp]=1

        big_bins=big_bins.astype('int')
        subset=[]
        lst=list(range(len(test)))
        initial=np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num=(big_bins[initial]>0).sum()
        cover_last=big_bins[initial]
        while True:
            flag=False
            for index in lst:
                temp1=np.bitwise_or(cover_last,big_bins[index])
                now_cover_num=(temp1>0).sum()
                if now_cover_num>max_cover_num:
                    max_cover_num=now_cover_num
                    max_index=index
                    max_cover=temp1
                    flag=True
            cover_last=max_cover
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            # print(max_cover_num)
        return subset

def kmnc_select(X, y, model, budget, k_bins=1000):
    layers = extract_layers(model)
    kmnc_model = kmnc(train=X, input=model.input, layers=layers, k_bins=k_bins)
    kmnc_model.fit(X)
    ranked_indices = kmnc_model.rank_fast(X)
    selected_indices = ranked_indices[:budget]
    return X[selected_indices], y[selected_indices], selected_indices

class nac(object):
    def __init__(self,test,input,layers,t=0):
        self.train=test
        self.input=input
        self.layers=layers
        self.t=t
        self.lst=[]
        self.neuron_activate=[]
        index_lst=[]

        for index,l in layers:
            self.lst.append(Model(inputs=input,outputs=l))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(test).reshape(len(test),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(test).reshape(len(test),l.shape[-1])
            temp=1/(1+np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_num=np.concatenate(self.neuron_activate,axis=1).shape[-1]
        self.lst=list(zip(index_lst,self.lst))

    def fit(self):
        neuron_activate=0
        for neu in self.neuron_activate:
            neuron_activate+=np.sum(np.sum(neu>self.t,axis=0)>0)
        return neuron_activate/float(self.neuron_num)

    def rank_fast(self,test):
        self.neuron_activate=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            temp=1/(1+np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate=np.concatenate(self.neuron_activate,axis=1)
        upper=(self.neuron_activate>self.t)

        subset=[]
        lst=list(range(len(test)))
        initial=np.random.choice(range(len(test)))
        lst.remove(initial)
        subset.append(initial)
        max_cover_num=np.sum(upper[initial])
        cover_last_1=upper[initial]
        while True:
            flag=False
            for index in lst:
                temp1=np.bitwise_or(cover_last_1,upper[index])
                cover1=np.sum(temp1)
                if cover1>max_cover_num:
                    max_cover_num=cover1
                    max_index=index
                    flag=True
                    max_cover1=temp1
            if not flag or len(lst)==1:
                break
            lst.remove(max_index)
            subset.append(max_index)
            cover_last_1=max_cover1
            # print(max_cover_num)
        return subset

    def rank_2(self,test):
        self.neuron_activate=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            temp=1/(1+np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate=np.concatenate(self.neuron_activate,axis=1)

        return np.argsort(np.sum(self.neuron_activate>self.t,axis=1))[::-1]

def nac_select(X, y, model, budget, t=0.5):
    layers = extract_layers(model)
    nac_model = nac(test=X, input=model.input, layers=layers, t=t)
    nac_model.fit()
    ranked_indices = nac_model.rank_fast(X)
    selected_indices = ranked_indices[:budget]
    return X[selected_indices], y[selected_indices], selected_indices

class LSA(object):
    def __init__(self,train,input,layers,std=0.05):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.lst=[]
        self.std_lst=[]
        self.mask=[]
        self.neuron_activate_train=[]

        for index, l in layers:
            self.lst.append((index, Model(inputs=input, outputs=l)))
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
            std_value = np.std(temp, axis=0)
            self.std_lst.append(std_value)
            self.mask.append(std_value > self.std)
        self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
        self.mask=np.concatenate(self.mask,axis=0)

    def fit(self,test,use_lower=False):
        self.neuron_activate_test=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test=np.concatenate(self.neuron_activate_test,axis=1)
        test_score = []
        # for test_sample in self.neuron_activate_test[:,self.mask]:
        #     test_mean = np.zeros_like(test_sample)
        #     for train_sample in self.neuron_activate_train[:,self.mask]:
        #         temp = test_sample-train_sample
        #         kde = gaussian_kde(temp, bw_method='scott')
        #         test_mean+=kde.evaluate(temp)
        #     test_score.append(reduce(lambda x,y:np.log(x)+np.log(y),test_mean/len(self.neuron_activate_train)))

        for test_sample in self.neuron_activate_test[:, self.mask]:
            deltas = self.neuron_activate_train[:, self.mask] - test_sample  # shape (n_train, n_features)
            deltas = deltas.T  # gaussian_kde expects shape (n_features, n_samples)
            kde = gaussian_kde(deltas, bw_method='scott')
            # numpy.linalg.LinAlgError: 18-th leading minor of the array is not positive definite
            # kde.covariance += np.eye(kde.covariance.shape[0]) * 1e-6
            test_score.append(np.log(kde.evaluate(np.zeros((deltas.shape[0], 1)))[0]))

        return test_score

def lsa_select(X, y, model, budget, std=0.05):
    layers = extract_layers(model)
    lsa_model = LSA(train=X, input=model.input, layers=layers, std=std)
    scores = lsa_model.fit(X)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

class DSA(object):
    def __init__(self,train,label,input,layers,std=0.05):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.lst=[]
        self.std_lst=[]
        self.mask=[]
        self.neuron_activate_train=[]
        index_lst=[]

        for index,l in layers:
            self.lst.append((index, Model(inputs=input, outputs=l)))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(train).reshape(len(train),l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
        self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
        # print('train_label shape', label.shape)
        self.train_label = to_ordinal(label)

    def fit(self,test,label,use_lower=False):
        self.neuron_activate_test=[]
        for index,l in self.lst:
            if index=='conv':
                temp=l.predict(test).reshape(len(test),-1,l.output.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=l.predict(test).reshape(len(test),l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test=np.concatenate(self.neuron_activate_test,axis=1)
        test_score = []
        # print('label shape', label.shape)
        label = to_ordinal(label)
        for test_sample,label_sample in zip(self.neuron_activate_test,label):
            dist_a = np.min(((self.neuron_activate_train[self.train_label == label_sample] - test_sample) ** 2).sum(axis=1))
            dist_b = np.min(((self.neuron_activate_train[self.train_label != label_sample] - test_sample) ** 2).sum(axis=1))
            test_score.append(dist_a/dist_b)
        return test_score

def dsa_select(X, y, model, budget, std=0.05):
    layers = extract_layers(model)
    dsa_model = DSA(train=X, label=y, input=model.input, layers=layers, std=std)
    scores = dsa_model.fit(X, y)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def geometric_diversity_select(X, y, model, budget, batch_size=128, layer_idx=-2, no_groups=50):
    min_max_scaler = preprocessing.MinMaxScaler()
    layer = model.layers[layer_idx]
    feat = []

    intermediate_layer_model = Model(inputs=model.input, outputs=layer.output)
    for x_batch, y_batch in make_batch(X, y, batch_size):
        _feat = intermediate_layer_model.predict(x_batch)
        feat.append(_feat)
    feat_mat = np.vstack(feat)
    # print(f"Feature matrix shape: {feat_mat.shape}")
    GD_scores, selected_indices = [], []
    for _ in range(no_groups):
        select_idx = np.random.choice(np.arange(len(feat_mat)), budget)
        selected_indices.append(select_idx)
        select_group = feat_mat[select_idx]
        # normalize group
        normalize_select_group = min_max_scaler.fit_transform(select_group)
        # compute GD
        GD = np.linalg.det(np.matmul(normalize_select_group, normalize_select_group.T))
        GD_scores.append(GD.squeeze())

        max_idx = np.argmax(np.array(GD_scores))
        chosen_indices = selected_indices[max_idx]
    
    return X[chosen_indices], y[chosen_indices], chosen_indices

'''
\textbf{Neuron Coverage (NC)} \cite{pei2017deepxplore} (coverage-based, 2017) computes the ratio of neurons in a given DNN $M$ that are activated above a self-defined threshold value by a given test suite $X_s$:  $NC(X_s) = \frac{|\{n| \forall x \in X_s, a(n,x)>t \}|}{|N|}$. $N = \{n_1, n_2,...\}$ denotes all neurons in the DNN model under test. $a(n,x)$ is the neuron activation value produced by test input $x$ on neuron $n \in N$. $t$ is the self-defined neuron activation threshold. We set $t=0.25$, which is commonly used in Deepxplore and DLFuzz. 

\textbf{Standard Deviation (STD)} \cite{aghababaeyan2023black} (diversity-based, 2023) is a statistical measure of how far from the mean a group of data points is. For a test suite $X_s$, it is calculated as the norm of the standard deviation of each feature in the input set. Formally, $STD(X_s) = \Vert (\sqrt{\sum_{i=1}^{n}  \frac{V_{x_{i,j}} - \mu_j}{n}}, 1 \leq j \leq m) \Vert$, where $V_x$ is the feature matrix of the input set $X_s$ , $m$ is the number of features, $\mu_j$ is the mean value of feature $j$ in $V_x$.
'''

def update_coverage(input_data, model, model_layer_dict, threshold):
    input_data = np.array(input_data)
    # 1. 找到所有需要统计的层
    outputs = []
    layer_names = []
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        outputs.append(layer.output)
        layer_names.append(layer.name)
    # 2. 用Model构造子模型
    feature_model = Model(inputs=model.input, outputs=outputs)
    # 3. 得到激活
    layer_outputs = feature_model.predict(input_data)
    if not isinstance(layer_outputs, list):
        layer_outputs = [layer_outputs]
    # 4. 每层每神经元统计
    for name, out in zip(layer_names, layer_outputs):
        neurons = out.shape[-1]
        for idx in range(neurons):
            act = out[(0,) + (slice(None),) * (out.ndim - 2) + (idx,)]
            activation = np.max(act) if isinstance(act, np.ndarray) else act
            if activation > threshold:
                model_layer_dict[(name, idx)] = True

def get_sample_coverage(x, model, base_layer_dict, threshold):
    # 复制 coverage 字典，不影响外部
    layer_dict = copy.deepcopy(base_layer_dict)
    update_coverage(x, model, layer_dict, threshold)
    covered = sum(layer_dict.values())
    return covered

def neuron_coverage_select(X, y, model, budget, threshold=0.75, batch_size=128):
    """
    Select data points that maximize neuron coverage.

    Args:
        X: Input data, shape [N, ...]
        y: Labels
        model: Keras model
        budget: Number of samples to select
        threshold: Activation threshold for coverage
        batch_size: Batch size for processing

    Returns:
        X_selected, y_selected, selected_indices
    """
    # 初始化覆盖表（所有神经元均未激活）
    base_layer_dict = defaultdict(bool)
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            base_layer_dict[(layer.name, index)] = False

    coverage_scores = []
    indices = []

    for batch_idx, (x_batch, y_batch) in enumerate(make_batch(X, y, batch_size)):
        for i in range(len(x_batch)):
            x = np.expand_dims(x_batch[i], axis=0)
            score = get_sample_coverage(x, model, base_layer_dict, threshold)
            coverage_scores.append(score)
            indices.append(batch_idx * batch_size + i)
    
    coverage_scores = np.array(coverage_scores)
    indices = np.array(indices)
    # 按 coverage 分数从高到低排序，选前 budget 个
    selected_idx = indices[np.argsort(coverage_scores)[::-1][:budget]]
    return X[selected_idx], y[selected_idx], selected_idx

def std_select(X, y, budget):
    X = np.asarray(X)
    if X.ndim > 2:
        # Flatten all but first axis (sample axis)
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    std_vec = np.std(X_flat, axis=0)
    std_norm = np.linalg.norm(std_vec)
    idx = np.argsort(std_norm)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def pace(X, y, model, budget, batch_size=128, layer_idx=-2, min_cluster_size=5, min_samples=5):
    import hdbscan
    """
    PACE testing selection metric.
    1. 提取模型在指定层的特征（默认倒数第二层）。
    2. 用 HDBSCAN 聚类，找出噪声点（异常点）。
    3. 优先选择噪声点，不足时再从最大簇采样。
    """
    # 1. 获取特征表示
    feature_model = Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    features = feature_model.predict(X, batch_size=batch_size)
    
    # 2. 归一化
    from sklearn.preprocessing import normalize
    features_norm = normalize(features)
    
    # 3. 聚类
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(features_norm)
    # -1 表示噪声点
    noise_idx = np.where(labels == -1)[0]
    normal_idx = np.where(labels != -1)[0]
    
    # 4. 选择样本
    if len(noise_idx) >= budget:
        selected_idx = noise_idx[:budget]
    else:
        # 不足预算时，补最大簇的样本
        from collections import Counter
        label_count = Counter(labels[normal_idx])
        # 找最大簇标签
        if label_count:
            max_cluster = label_count.most_common(1)[0][0]
            cluster_idx = normal_idx[labels[normal_idx] == max_cluster]
            n_needed = budget - len(noise_idx)
            extra_idx = cluster_idx[:n_needed]
            selected_idx = np.concatenate([noise_idx, extra_idx])
        else:
            selected_idx = noise_idx  # 只有噪声点
    
    # 5. 返回被选中的样本
    return X[selected_idx], y[selected_idx], selected_idx

def dr(X, y, model, budget, batch_size=128, KN=20):
    """
    DeepReduce-inspired DNN test selection metric function.
    
    Args:
        X: Input data, shape (N, ...).
        y: Labels for X.
        model: Keras model with a .predict method, outputs neuron activations (not just softmax).
        budget: Number of samples to select.
        batch_size: Batch size for model predictions.
        KN: Number of bins for neuron activation quantization.
        
    Returns:
        X_selected, y_selected, selected_idx (np.ndarray)
    """
    # 1. Get model outputs (activations) for all data
    outputs = []
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        batch_out = model.predict(x_batch)
        # If model outputs (N, C), reshape to (N, C)
        outputs.append(batch_out)
    outputs = np.concatenate(outputs, axis=0)  # Shape (N, num_neurons)
    tnum, nnum = outputs.shape

    # 2. Quantize activations (heavy-tailed binning, like getSection_heavytailed)
    def quantize_heavytailed(outputs, KN):
        NIndex = np.zeros_like(outputs, dtype=int)
        for n in range(outputs.shape[1]):
            values = outputs[:, n]
            uniq = np.unique(values)
            uniq.sort()
            thresholds = [uniq[int(len(uniq)/KN*i)] for i in range(KN)]
            thresholds.append(uniq[-1] + 1e-6)
            for i in range(len(values)):
                for k in range(KN):
                    if values[i] < thresholds[k+1]:
                        NIndex[i, n] = k
                        break
        return NIndex
    NIndex = quantize_heavytailed(outputs, KN)

    # 3. Compute neuron activation distribution over dataset (getDistribute_heavytailed)
    def calc_dist(NIndex, KN):
        NDist = np.zeros((NIndex.shape[1], KN))
        for i in range(NIndex.shape[0]):
            for n in range(NIndex.shape[1]):
                NDist[n, NIndex[i, n]] += 1
        NDist /= NIndex.shape[0]
        return NDist
    NDist = calc_dist(NIndex, KN)

    # 4. Score each sample by its KL divergence to the global neuron distribution (like calculate_kl)
    def kl_score(sample_idx):
        # For a sample, get bin indices for all neurons
        bins = NIndex[sample_idx]
        p = np.zeros((nnum, KN))
        for i in range(nnum):
            p[i, bins[i]] = 1.0
        eps = 1e-15
        kl = np.sum(p * np.log(np.clip(p / np.clip(NDist, eps, None), eps, None)))
        return kl

    scores = np.array([kl_score(i) for i in range(tnum)])

    # 5. Select top-k by KL score (high KL = more "unusual" sample for neuron coverage)
    idx = np.argsort(scores)[::-1]  # largest KL first
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def neuron_coverage_score(model, X, neuron_interval, neuron_proba, num_neurons, batch_size=128, layer_idx=-3):
    """
    Calculate coverage-based entropy score for each sample in X.
    The score reflects how much each sample deviates from the reference neuron output distribution.
    """
    layer = model.layers[layer_idx]
    # num_neurons = neuron_proba[(layer.name, 0)].shape[0] if (layer.name, 0) in neuron_proba else 0
    coverage_scores = []
    # Precompute reference distribution to avoid recalculation
    ref_proba = {idx: neuron_proba[(layer.name, idx)] for idx in range(num_neurons)}
    intervals = {idx: neuron_interval[(layer.name, idx)] for idx in range(num_neurons)}
    for x_batch, _ in make_batch(X, np.zeros(X.shape[0]), batch_size):
        # Get output for batch
        func = K.function([model.input], [layer.output])
        output = func([x_batch])[0].reshape(x_batch.shape[0], -1)
        batch_scores = []
        for i in range(output.shape[0]):
            sample_scores = []
            for idx in range(output.shape[1]):
                # Place sample output into intervals
                interval = intervals[idx]
                ref = ref_proba[idx]
                val = output[i, idx]
                # Find which bin val lands in
                bin_idx = np.searchsorted(interval, val, side='right') - 1
                bin_idx = np.clip(bin_idx, 0, len(ref) - 1)
                # Score: negative log-probability for that bin (i.e., "surprisal")
                p = np.clip(ref[bin_idx], 1e-10, 1.0)
                sample_scores.append(-np.log(p))
            batch_scores.append(np.mean(sample_scores))  # Use mean surprisal as sample's coverage score
        coverage_scores.extend(batch_scores)
    return np.array(coverage_scores)

def build_neuron_tables_for_ces(model, X, divide=10, batch_size=128, layer_idx=-3):
    """
    Build neuron interval and probability tables for reference.
    """
    layer = model.layers[layer_idx]
    func = K.function([model.input], [layer.output])
    outputs = []
    for x_batch, _ in make_batch(X, np.zeros(X.shape[0]), batch_size):
        out = func([x_batch])[0].reshape(x_batch.shape[0], -1)
        outputs.append(out)
    output = np.concatenate(outputs, axis=0)
    neuron_interval = {}
    neuron_proba = {}
    total_num = output.shape[0]
    num_neurons = output.shape[1]
    for idx in range(num_neurons):
        lower, upper = np.min(output[:, idx]), np.max(output[:, idx])
        interval = np.linspace(lower, upper, divide + 1)
        hist, _ = np.histogram(output[:, idx], bins=interval)
        proba = hist / total_num
        neuron_interval[(layer.name, idx)] = interval
        neuron_proba[(layer.name, idx)] = proba
    return neuron_interval, neuron_proba, num_neurons

def ces_select(X, y, model, budget, batch_size=128, layer_idx=-3, divide=10):
    neuron_interval, neuron_proba, num_neurons = build_neuron_tables_for_ces(model, X, divide, batch_size, layer_idx)
    coverage_scores = neuron_coverage_score(model, X, neuron_interval, neuron_proba, num_neurons, batch_size, layer_idx)
    idx = np.argsort(coverage_scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def mcp_score(proba):
    """
    MCP分数：次大值与最大值的比值，越高表示越不自信
    proba: 2D numpy array, shape (n_samples, n_classes)
    """
    sorted_proba = np.sort(proba, axis=1)
    max_proba = sorted_proba[:, -1]
    second_max_proba = sorted_proba[:, -2]
    score = second_max_proba / (max_proba + 1e-8)
    return score

def mcp_select(X, y, model, budget, batch_size=128):
    """
    按MCP度量选择测试用例
    X: 输入样本
    y: 样本标签
    model: Keras模型
    budget: 选择多少个样本
    batch_size: 批量大小
    返回：被选中的 X, y, 以及被选中的下标
    """
    mcp_scores = []
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        proba = model.predict(x_batch)
        mcp_scores.append(mcp_score(proba))
    scores = np.concatenate(mcp_scores)
    idx = np.argsort(scores)[::-1]  # 按最大MCP分数降序
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def deepest(X, y, model, budget, batch_size=128, occurrence_prob=None):
    """
    DeepEST selection metric for DNN testing.

    X: np.ndarray, input data
    y: np.ndarray, labels
    model: Keras model with .predict()
    budget: int, number of samples to select
    occurrence_prob: None or np.ndarray, optional, shape (n_samples,)
        If None, uniform probability is used.
    d: float, probability threshold for selection (between 0 and 1)
    batch_size: int, for model prediction

    Returns: (X_sel, y_sel, selected_idx)
    """
    n_samples = X.shape[0]
    if occurrence_prob is None:
        occurrence_prob = np.ones(n_samples) / n_samples

    # 1. Model prediction
    proba = []
    for i in range(0, n_samples, batch_size):
        proba.append(model.predict(X[i:i+batch_size]))
    proba = np.concatenate(proba, axis=0)

    # 2. 假设“失败概率”为 1 - max proba（最高的不确定性/出错可能）
    fail_prob = 1 - np.max(proba, axis=1)

    # 3. DeepEST 估计分数: occurrence_prob * fail_prob
    estimationX = occurrence_prob * fail_prob * n_samples

    # 4. 选择分数最高的样本
    idx = np.argsort(estimationX)[::-1]  # 从高到低排序
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def select(X, y, model, budget, metric, batch_size=128, **kwargs):
    if metric == 'rnd':
        return random_select(X, y, budget)
    elif metric == 'ent':
        return entropy_select(X, y, model, budget, batch_size)
    elif metric == 'gini':
        return gini_select(X, y, model, budget, batch_size)
    elif metric == 'dat':
        return dat_ood_detector(X, y, model, budget, trainX=kwargs.get('trainX', None), trainy=kwargs.get('trainy', None), hybridX=kwargs.get('hybridX', None), hybridy=kwargs.get('hybridy', None), batch_size=batch_size, num_classes=kwargs.get('num_classes', None))
    elif metric == 'kmnc':
        return kmnc_select(X, y, model, budget, k_bins=kwargs.get('k_bins', 1000))
    elif metric == 'nac':
        return nac_select(X, y, model, budget, t=kwargs.get('t', 0.5))
    elif metric == 'lsa':
        return lsa_select(X, y, model, budget, std=kwargs.get('std', 0.05))
    elif metric == 'dsa':
        return dsa_select(X, y, model, budget, std=kwargs.get('std', 0.05))
    elif metric == 'gd':
        return geometric_diversity_select(X, y, model, budget, batch_size, layer_idx=kwargs.get('layer_idx', -2), no_groups=kwargs.get('no_groups', 50))
    elif metric == 'nc':
        return neuron_coverage_select(X, y, model, budget, threshold=kwargs.get('threshold', 0.75), batch_size=batch_size)
    elif metric == 'std':
        return std_select(X, y, budget)
    elif metric == 'pace':
        return pace(X, y, model, budget, batch_size=batch_size, layer_idx=kwargs.get('layer_idx', -2), min_cluster_size=kwargs.get('min_cluster_size', 5), min_samples=kwargs.get('min_samples', 5))
    elif metric == 'dr':
        return dr(X, y, model, budget, batch_size=batch_size, KN=kwargs.get('KN', 20))
    elif metric == 'ces':
        return ces_select(X, y, model, budget, batch_size=batch_size, layer_idx=kwargs.get('layer_idx', -3), divide=kwargs.get('divide', 10))
    elif metric == 'mcp':
        return mcp_select(X, y, model, budget, batch_size=batch_size)
    elif metric == 'est':
        return deepest(X, y, model, budget, batch_size=batch_size, occurrence_prob=kwargs.get('occurrence_prob', None))
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")