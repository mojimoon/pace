'''
rewrite of selection_metrics.py in flavor of Keras
'''

from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from scipy.stats import gaussian_kde
from functools import reduce

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
    return X[selected_idx], y[selected_idx], idx

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
    return X[selected_idx], y[selected_idx], idx

def extract_layers(model):
    layers = []
    for l in model.layers:
        if isinstance(l, Conv2D):
            layers.append(('conv', l.output))
        elif isinstance(l, Dense):
            layers.append(('dense', l.output))
    return layers

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
            self.neuron_activate_train.append(temp.copy())
            self.std_lst.append(np.std(temp,axis=0))
            self.mask.append((np.array(self.std_lst)>std))
        self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
        self.mask=np.concatenate(self.mask,axis=0)
        #self.lst=list(zip(index_lst,self.lst))

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
        for test_sample in self.neuron_activate_test[:,self.mask]:
            test_mean = np.zeros_like(test_sample)
            for train_sample in self.neuron_activate_train[:,self.mask]:
                temp = test_sample-train_sample
                kde = gaussian_kde(temp, bw_method='scott')
                test_mean+=kde.evaluate(temp)
            test_score.append(reduce(lambda x,y:np.log(x)+np.log(y),test_mean/len(self.neuron_activate_train)))
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
            self.lst.append(Model(inputs=input,outputs=l))
            index_lst.append(index)
            i=Model(inputs=input,outputs=l)
            if index=='conv':
                temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                temp=np.mean(temp,axis=1)
            if index=='dense':
                temp=i.predict(train).reshape(len(train),l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
        self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
        self.train_label = np.array(label)

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
        for test_sample,label_sample in zip(self.neuron_activate_test,label):
            dist_a = np.min(((self.neuron_activate_train[self.train_label == label_sample,:]-test_sample)**2).sum(axis=1))
            dist_b = np.min(((self.neuron_activate_train[self.train_label != label_sample,:]-test_sample)**2).sum(axis=1))
            test_score.append(dist_a/dist_b)
        return test_score

def dsa_select(X, y, model, budget, std=0.05):
    layers = extract_layers(model)
    dsa_model = DSA(train=X, label=y, input=model.input, layers=layers, std=std)
    scores = dsa_model.fit(X, y)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def geometric_diversity_select(X, y, model, budget, batch_size=128, layer_idx=None, no_groups=50):
    min_max_scaler = preprocessing.MinMaxScaler()

    if layer_idx is not None:
        layer = model.layers[layer_idx]
    else:
        layer = model.layers[-2] # typically the last dense layer before output

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

def predict_activations(model, X, layer_names=None, batch_size=128):
    if layer_names is None:
        # select all Dense and Aactivation layers
        selected_layers = [l for l in model.layers if isinstance(l, Dense) or 'activation' in l.name]
    else:
        selected_layers = [l for l in model.layers if l.name in layer_names]
    intermediate_layer_model = Model(inputs=model.input, outputs=[l.output for l in selected_layers])
    acts = intermediate_layer_model.predict(X, batch_size=batch_size)
    if isinstance(acts, list):
        acts = [a.reshape((a.shape[0], -1)) for a in acts]
        activations = np.concatenate(acts, axis=1)
    else:
        activations = acts
    return activations

def neuron_coverage_select(X, y, model, budget, t=0.25, batch_size=128):
    activations = predict_activations(model, X, batch_size=batch_size)
    neuron_max = np.max(activations, axis=0) # shape (num_neurons,)
    covered = (neuron_max > t)
    nc = np.sum(covered) / covered.size

    idx = np.argsort(nc)[::-1]
    selected_idx = idx[:budget]
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

def select(X, y, model, budget, metric, batch_size=128, **kwargs):
    if metric == 'rnd':
        return random_select(X, y, budget)
    elif metric == 'ent':
        return entropy_select(X, y, model, budget, batch_size)
    elif metric == 'gini':
        return gini_select(X, y, model, budget, batch_size)
    elif metric == 'kmnc':
        return kmnc_select(X, y, model, budget, k_bins=kwargs.get('k_bins', 1000))
    elif metric == 'nac':
        return nac_select(X, y, model, budget, t=kwargs.get('t', 0.5))
    elif metric == 'lsa':
        return lsa_select(X, y, model, budget, std=kwargs.get('std', 0.05))
    elif metric == 'dsa':
        return dsa_select(X, y, model, budget, std=kwargs.get('std', 0.05))
    elif metric == 'gd':
        return geometric_diversity_select(X, y, model, budget, batch_size, layer_idx=kwargs.get('layer_idx'), no_groups=kwargs.get('no_groups', 50))
    elif metric == 'nc':
        return neuron_coverage_select(X, y, model, budget, t=kwargs.get('t', 0.25), batch_size=batch_size)
    elif metric == 'std':
        return std_select(X, y, budget)
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")