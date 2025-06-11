'''
rewrite of selection_metrics.py in flavor of Keras
'''

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
    n_samples = X.shape[0]
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
    n_samples = X.shape[0]
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
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")