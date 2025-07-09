'''
rewrite of selection_metrics.py in flavor of Keras
'''
import os.path
from collections import defaultdict
from sklearn import preprocessing
import tensorflow as tf
from keras import Input, Sequential
import numpy as np
from keras.models import Model
# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Dense
import keras.layers as klayers
from scipy.stats import gaussian_kde
from functools import reduce
import copy
import keras.backend as K
import subprocess
import tempfile
from pathlib import Path
from typing import List, Any
import sys
import random

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
    return X[selected_idx], y[selected_idx], selected_idx

def gini_select_score(X, y, model, budget, batch_size=128):
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
        if isinstance(l, klayers.Conv2D):
            layers.append(('conv', l.output))
        elif isinstance(l, klayers.Dense):
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
        if isinstance(layer, klayers.Dense):
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
    
    _, _, _, id_scores = gini_select_score(canX_id, cany_id, model, id_select_num, batch_size)
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

def nac_select(X, y, model, budget, t=0.75):
    layers = extract_layers(model)
    nac_model = nac(test=X, input=model.input, layers=layers, t=t)
    nac_model.fit()
    ranked_indices = nac_model.rank_2(X)
    selected_indices = ranked_indices[:budget]
    return X[selected_indices], y[selected_indices], selected_indices

class LSA(object):
    def __init__(self,train,input,layers,dataset_name,model_name,std=0.05):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=input
        self.layers=layers
        self.std=std
        self.name=dataset_name
        self.model_name = model_name
        self.lst=[]
        self.std_lst=[]
        self.mask=[]
        self.neuron_activate_train=[]

        for index,l in [layers[-2]]:
            self.lst.append((index, Model(inputs=input, outputs=l)))
            i = Model(inputs=input, outputs=l)
            if 'mnist' in dataset_name:
                trainset_name = 'mnist'
            path = f'./AllResult/LSA/{trainset_name}_{model_name}_neuron_activate_train.npy'
            if os.path.exists(path):
                self.neuron_activate_train = np.load(path)
            else:
                if index == 'conv':
                    temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                    temp = np.mean(temp, axis=1)
                if index == 'dense':
                    temp = i.predict(train).reshape(len(train), l.shape[-1])
                self.neuron_activate_train.append(temp.copy())
                self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)
                np.save(path, self.neuron_activate_train)
            std_value = np.std(self.neuron_activate_train, axis=0)
            self.std_lst.append(std_value)
            self.mask.append(std_value > self.std)

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
            # deltas += 1e-6 * np.random.randn(*deltas.shape)
            kde = gaussian_kde(deltas, bw_method='scott')
            test_score.append(np.log(kde.evaluate(np.zeros((deltas.shape[0], 1)))[0]))

        return test_score

def lsa_select(trainX, trainy, X, y, model, budget, dataset, model_name, std=0.05):
    layers = extract_layers(model)
    lsa_model = LSA(train=trainX, input=model.input, layers=layers, dataset_name=dataset, model_name=model_name, std=std)
    scores = lsa_model.fit(X)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

class DSA(object):
    def __init__(self,train,label,model,layers,dataset_name,model_name,std=0.05):
        '''
        train:训练集数据
        input:输入张量
        layers:输出张量层
        '''
        self.train=train
        self.input=model.input
        self.model=model
        self.layers=layers
        #self.std=std
        self.model_name=model_name
        self.lst=[]
        self.conf=[]
        self.std_lst=[]
        self.mask=[]
        self.neuron_activate_train=[]
        self.name=dataset_name
        index_lst=[]

        # we use the last hidden layer for all models (NOT softmax).
        for index,l in [layers[-2]]:
            self.lst.append((index, Model(inputs=model.input, outputs=l)))
            index_lst.append(index)
            i=Model(inputs=model.input,outputs=l)
            if 'mnist' in dataset_name:
                trainset_name = 'mnist'
            elif 'udacity' in dataset_name:
                trainset_name = 'udacity'
            path = f'./AllResult/DSA/{trainset_name}_{model_name}_neuron_activate_train.npy'
            if os.path.exists(path):
                self.neuron_activate_train = np.load(path)
            else:
                if index=='conv':
                    temp=i.predict(train).reshape(len(train),-1,l.shape[-1])
                    print('temp shape', temp.shape)
                    temp=np.mean(temp,axis=1)
                if index=='dense':
                    temp=i.predict(train).reshape(len(train),l.shape[-1])
                self.neuron_activate_train.append(temp.copy())
                self.neuron_activate_train=np.concatenate(self.neuron_activate_train,axis=1)
                np.save(path, self.neuron_activate_train)
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
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)
        # save confidence score fort deepest
        for _, conf_l in [self.layers[-1]]:
            conf_layer = Model(inputs=self.model.input, outputs=conf_l)
            temp_conf = conf_layer.predict(test).reshape(len(test), conf_layer.output.shape[-1])
            softmax = tf.nn.softmax(temp_conf)
        test_score = []
        # print('label shape', label.shape)
        label = to_ordinal(label)
        for test_sample,label_sample in zip(self.neuron_activate_test,label):
            dist_a = np.min(((self.neuron_activate_train[self.train_label == label_sample] - test_sample) ** 2).sum(axis=1))
            dist_b = np.min(((self.neuron_activate_train[self.train_label != label_sample] - test_sample) ** 2).sum(axis=1))
            test_score.append(dist_a/dist_b)
        # save DSA value
        breakpoint()
        np.save(f'./AllResult/DSA/{self.name}_{self.model_name}_dsa.npy', np.array(test_score))
        np.save(f'./AllResult/DSA/{self.name}_{self.model_name}_confidence.npy', np.array(softmax))
        return test_score

def dsa_select(trainX, trainy, X, y, model, budget, dataset, model_name, std=0.05):
    layers = extract_layers(model)
    dsa_model = DSA(train=trainX, label=trainy, model=model, layers=layers, dataset_name=dataset, model_name=model_name, std=std)
    breakpoint()
    scores = dsa_model.fit(X, y)
    # save normalized dsa score [0,1] for deepest

    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:budget]
    return X[selected_idx], y[selected_idx], selected_idx

def geometric_diversity_select(X, y, dataset_name, budget, batch_size=128, no_groups=50):
    min_max_scaler = preprocessing.MinMaxScaler()
    feat = []

    if dataset_name.lower() in ['mnist', 'udacity']:
        vgg16_imagenet_path = '/home/jzhang2297/empirical/pace/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        feature_extractor = tf.keras.applications.VGG16(include_top=False, weights=vgg16_imagenet_path)
        layer = feature_extractor.layers[-1]  # maxpooling layer after the last convolutional layer.
        intermediate_layer_model = Model(inputs=feature_extractor.input, outputs=layer.output)

    elif dataset_name.lower() in ['androzoo']:
        from learner import model_scope_dict
        model = 'deepdrebin'
        targeted_model_names_dict = model_scope_dict.copy()
        targeted_model = targeted_model_names_dict[model](mode='test')
        ckpt_dir = '/home/jzhang2297/anomaly/malware/adversarial-deep-ensemble-droidmawlare/{0}/saved_parameters/{1}/'.format(dataset_name, model)
        sess = tf.Session()
        cur_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        layer = targeted_model.dense1
    else:
        layer = None
    breakpoint()
    # intermediate_layer_model = Model(inputs=Input(shape=X[0].shape), outputs=layer.output)
    for x_batch, y_batch in make_batch(X, y, batch_size):
        if x_batch.shape[-1] == 1:
            x_batch = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_batch, dtype=tf.float32))
            if x_batch.shape[1] < 32 or x_batch.shape[2] < 32:
                x_batch = tf.image.resize(x_batch, [32, 32])
        _feat = intermediate_layer_model.predict(x_batch)  # MNIST (32,32,3) -> (1,1,512) feature shape
        feat.append(_feat.reshape(_feat.shape[0],-1))
    feat_mat = np.vstack(feat)
    print(f"Feature matrix shape: {feat_mat.shape}")
    breakpoint()
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

def std_select(X, y, dataset_name, budget, batch_size=128, no_groups=50):
    min_max_scaler = preprocessing.MinMaxScaler()
    feat = []

    if dataset_name.lower() in ['mnist', 'udacity']:
        vgg16_imagenet_path = '/home/jzhang2297/empirical/pace/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        feature_extractor = tf.keras.applications.VGG16(include_top=False, weights=vgg16_imagenet_path)
        layer = feature_extractor.layers[-1]  # maxpooling layer after the last convolutional layer.
        intermediate_layer_model = Model(inputs=feature_extractor.input, outputs=layer.output)

    elif dataset_name.lower() in ['androzoo']:
        from learner import model_scope_dict
        model = 'deepdrebin'
        targeted_model_names_dict = model_scope_dict.copy()
        targeted_model = targeted_model_names_dict[model](mode='test')
        ckpt_dir = '/home/jzhang2297/anomaly/malware/adversarial-deep-ensemble-droidmawlare/{0}/saved_parameters/{1}/'.format(
            dataset_name, model)
        sess = tf.Session()
        cur_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        layer = targeted_model.dense1
    else:
        layer = None

    # intermediate_layer_model = Model(inputs=Input(shape=X[0].shape), outputs=layer.output)
    for x_batch, y_batch in make_batch(X, y, batch_size):
        if x_batch.shape[-1] == 1:
            x_batch = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_batch, dtype=tf.float32))
            if x_batch.shape[1]<32 or x_batch.shape[2]<32:
                x_batch = tf.image.resize(x_batch, [32,32])
        _feat = intermediate_layer_model.predict(x_batch) # MNIST (32,32,3) -> (1,1,512) feature shape
        feat.append(_feat.reshape(_feat.shape[0],-1))
    feat_mat = np.vstack(feat)
    print(f"Feature matrix shape: {feat_mat.shape}")
    breakpoint()
    std_scores, selected_indices = [], []
    for _ in range(no_groups):
        select_idx = np.random.choice(np.arange(len(feat_mat)), budget)
        selected_indices.append(select_idx)
        select_group = feat_mat[select_idx]
        # normalize group
        normalize_select_group = min_max_scaler.fit_transform(select_group)
        # compute STD
        STD = np.linalg.norm(np.std(normalize_select_group, axis=0))
        std_scores.append(STD.squeeze())
    max_idx = np.argmax(np.array(std_scores))
    chosen_indices = selected_indices[max_idx]
    return X[chosen_indices], y[chosen_indices], chosen_indices

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

class ActiveSet:
    def __init__(self, N, T, occurrenceP):
        self.testFrame = []
        self.id = [0] * N
        self.outcome = [False] * N
        self.weights = [0.0] * T
        self.occurrenceProb = list(occurrenceP)
        self.outcomeSum = 0
        self.outcomeSumX = 0.0
        self.qi = 0.0

    def getOutcomeSum(self):
        return self.outcomeSum

    def getOutcomeSumX(self):
        return self.outcomeSumX

    def getWeights(self, k):
        return self.weights[k]

    def activeSetUpdate(self, tF, tFnumber, y, upweights):
        self.testFrame.append(tF)
        self.id[len(self.testFrame) - 1] = tFnumber
        self.outcome[len(self.testFrame) - 1] = y

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += upweights[tFnumber][i]

        # Zero out weights of selected test frames
        for i in range(len(self.testFrame)):
            self.weights[self.id[i]] = 0.0

        if y:
            self.outcomeSum += 1
            self.outcomeSumX += self.occurrenceProb[tFnumber]

    def activeSetUpdateExp(self, tF, tFnumber, y, upweights):
        self.testFrame.append(tF)
        self.id[len(self.testFrame) - 1] = tFnumber

        for i in range(len(self.weights)):
            self.weights[i] += upweights[tFnumber][i]

        for i in range(len(self.testFrame)):
            self.weights[self.id[i]] = 0.0

        if y > 0:
            self.outcomeSumX += self.occurrenceProb[tFnumber] * y

    def testFrameExtraction(self, d):
        total_weight = sum(self.weights)
        rand = random.random()
        k = 0
        cumulative_prob = self.weights[0] / total_weight if total_weight != 0 else 0

        while k < len(self.weights) - 1 and rand >= cumulative_prob:
            k += 1
            cumulative_prob += self.weights[k] / total_weight if total_weight != 0 else 0

        if total_weight != 0:
            self.qi = d * (self.weights[k] / total_weight) + (1 - d) * (1 / (len(self.weights) - len(self.testFrame)))
        else:
            self.qi = 1 / (len(self.weights) - len(self.testFrame))

        return k

    def qiCalculation(self, d, k):
        total_weight = sum(self.weights)
        if total_weight != 0:
            self.qi = d * (self.weights[k] / total_weight) + (1 - d) * (1 / (len(self.weights) - len(self.testFrame)))
        else:
            self.qi = 1 / (len(self.weights) - len(self.testFrame))

    def printSelectedTestFrame(self):
        print("Selected Test Frame are:", self.testFrame)
class TestFrame:
    def __init__(self, name, tfID, failureProb, occurrenceProb, output, fail):
        self.name = name
        self.tfID = tfID
        self.failureProb = failureProb
        self.occurrenceProb = occurrenceProb
        self.output = output
        self.fail = fail

    def extractAndExecuteTestCase(self):
        # Returns True if this test frame simulates a failure
        return self.fail

    def getOutput(self):
        return self.output

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getTfID(self):
        return self.tfID

    def setTfID(self, tfID):
        self.tfID = tfID

    def getFailureProb(self):
        return self.failureProb

    def setFailureProb(self, failureProb):
        self.failureProb = failureProb

    def getOccurrenceProb(self):
        return self.occurrenceProb

    def setOccurrenceProb(self, occurrenceProb):
        self.occurrenceProb = occurrenceProb
class DeepESTSelector:
    def __init__(self):
        self.numfp = 0
        self.z = []
        self.failedRequest = []

    def getZ(self):
        return self.z

    def getnumfp(self):
        return self.numfp

    def getfailedRequest(self):
        return self.failedRequest

    def selectAndRunTestCase(self, n, testFrameList, weightsMatrix, d):
        self.failedRequest.clear()
        self.numfp = 0

        if n > len(testFrameList) or n <= 0:
            return [-1.0, -1.0]

        if d <= 0 or d >= 1:
            return [-2.0, -2.0]

        scompl = list(range(len(testFrameList)))
        occurrenceProb = [tf.getOccurrenceProb() for tf in testFrameList]

        randomNum = random.randint(0, len(testFrameList) - 1)
        ak = ActiveSet(n, len(testFrameList), occurrenceProb)

        name = testFrameList[randomNum].getTfID()
        esito = testFrameList[randomNum].extractAndExecuteTestCase()
        tc = testFrameList[randomNum].getName()

        ak.activeSetUpdate(name, randomNum, esito, weightsMatrix)
        scompl.remove(randomNum)

        if esito:
            y = 1
            self.numfp += 1
            self.failedRequest.append(tc)
        else:
            y = 0

        estimationX = [0] * n
        estimationX[0] = len(testFrameList) * (occurrenceProb[randomNum] * y)

        k = 1
        while k < n:
            weightsSum = sum(ak.getWeights(i) for i in scompl)

            if weightsSum == 0:
                prob = d + 0.1
            else:
                prob = random.random()

            if prob <= d: # WBS
                current_tf = ak.testFrameExtraction(d)
            else: # random selection SRS
                random_idx = random.randint(0, len(scompl) - 1)
                current_tf = scompl[random_idx]

            name = testFrameList[current_tf].getTfID()
            esito = testFrameList[current_tf].extractAndExecuteTestCase()
            tc = testFrameList[current_tf].getName()

            ziX = ak.getOutcomeSumX()

            if esito:
                if prob > d:
                    ak.qiCalculation(d, current_tf)
                ziX += occurrenceProb[current_tf] / ak.qi
                self.numfp += 1
                self.failedRequest.append(tc)

            ak.activeSetUpdate(name, current_tf, esito, weightsMatrix)

            if prob <= d:
                scompl.remove(current_tf)
            else:
                scompl.pop(random_idx)

            estimationX[k] = ziX
            k += 1
        selected_test_suite = ak.testFrame
        return selected_test_suite
        #self.z = estimationX
        #return self.estimatorBoCSP(n, estimationX)

    def estimatorBoCSP(self, n, estimationX):
        sumX = sum(estimationX)
        mean = sumX / n

        variance = sum((estimationX[i] - estimationX[0]) ** 2 for i in range(1, n)) / (n * (n - 1))
        return [mean, variance]

class TestCase:
    string_ok = ""
    string_not_ok = "different results"

    class ExecutionState:
        EXECUTED = "executed"
        NOT_EXECUTED = "notExecuted"

    def __init__(self, name: str, tcID: str, root_directory: Path = None):
        self.name = name
        self.tcID = tcID
        self.outcome = False
        self.execution_state = self.ExecutionState.NOT_EXECUTED
        self.path_root_directory = root_directory or Path.cwd()
        self.number_of_commands = 0
        self.list_of_commands: List[str] = []
        self.inputs: List[Any] = []
        self.max_number_of_inputs = 0
        self.expected_occurrence_probability = 0.0
        self.real_occurrence_probability = 0.0
        self.expected_failure_likelihood = 0.0

    def __eq__(self, other):
        if not isinstance(other, TestCase):
            return False
        return self.tcID == other.tcID

    def __hash__(self):
        return hash(self.name)

    def run_test_case_dummy(self, string: str) -> bool:
        """Simulates execution."""
        self.outcome = self.get_outcome()
        self.execution_state = self.ExecutionState.EXECUTED
        return self.outcome

    def run_test_case(self) -> bool:
        """Actually runs the shell script and checks output."""
        script_path = self._create_temp_script()

        print(f"\n\n***** Running Test {self.name}. ******\nExecuted Temporary Script: {script_path}")
        try:
            result = subprocess.run(
                [script_path],
                cwd=self.path_root_directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            print(f"\n'Standard Output' Printed by the test case execution {self.name}: {stdout}")
            if result.returncode != 0:
                print(f"\n'Standard Error' Printed by the test case execution {self.name}: {stderr}")
                print(f"Execution State: {result.returncode}")
                self.execution_state = self.ExecutionState.NOT_EXECUTED
            else:
                self.execution_state = self.ExecutionState.EXECUTED

            if stdout == self.string_not_ok:
                self.outcome = False
            elif stdout == self.string_ok:
                self.outcome = True

        finally:
            script_path.unlink()  # delete the temp script

        print(f"\n ** Test {self.name} terminated **\n")
        return self.outcome

    def _create_temp_script(self) -> Path:
        """Create a temporary bash script with the list of commands."""
        fd, path = tempfile.mkstemp(suffix=".sh", dir=self.path_root_directory)
        script_path = Path(path)
        script_path.chmod(0o755)

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            for cmd in self.list_of_commands:
                f.write(cmd + "\n")

        return script_path

    # --- Getter / Setter properties ---

    def get_name(self):
        return self.name

    def set_name(self, name: str):
        self.name = name

    def get_number(self):
        return self.tcID

    def set_number(self, number: str):
        self.tcID = number

    def get_number_of_commands(self):
        return self.number_of_commands

    def set_number_of_commands(self, num: int):
        self.number_of_commands = num

    def get_list_of_commands(self):
        return self.list_of_commands

    def set_list_of_commands(self, cmds: List[str]):
        self.list_of_commands = cmds

    def get_expected_occurrence_probability(self):
        return self.expected_occurrence_probability

    def set_expected_occurrence_probability(self, value: float):
        self.expected_occurrence_probability = value

    def get_real_occurrence_probability(self):
        return self.real_occurrence_probability

    def set_real_occurrence_probability(self, value: float):
        self.real_occurrence_probability = value

    def set_outcome(self, outcome: bool):
        self.outcome = outcome

    def get_outcome(self) -> bool:
        return self.outcome

    def get_inputs(self):
        return self.inputs

    def set_inputs(self, inputs: List[Any]):
        self.inputs = inputs

    def get_max_number_of_inputs(self):
        return self.max_number_of_inputs

    def set_max_number_of_inputs(self, value: int):
        self.max_number_of_inputs = value

    def get_tcID(self):
        return self.tcID

    def set_tcID(self, tcID: str):
        self.tcID = tcID

    def get_execution_state(self):
        return self.execution_state

    def set_execution_state(self, state: str):
        self.execution_state = state

class InitializerTF:
    def __init__(self, path: str):
        self.csv_file = path

    def readTestFrames(self, key: int, size: int) -> List[TestFrame]:
        test_frames = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                header = next(f)  # Skip header
                occ = 1.0 / size  # Uniform occurrence probability

                for i, line in enumerate(f):
                    parts = line.strip().split(",")
                    outcome = parts[1]
                    fail = outcome != "Pass"
                    val = float(parts[key])

                    if key in [3, 6]:  # invert confidence and combo
                        val = 1.0 - val
                        # optional epsilon:
                        # if val == 0.0:
                        #     val = 1e-9

                    tf = TestFrame(
                        name=str(len(test_frames)),
                        tfID=str(len(test_frames)),
                        failureProb=val,
                        occurrenceProb=occ,
                        output=parts[2],
                        fail=fail
                    )
                    test_frames.append(tf)

                if i + 1 != size:
                    print("[WARNING] The size is lower/greater!!!")

        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {e}")
        except Exception as e:
            print(f"[ERROR] {e}")

        return test_frames

    def weightedMatrixComputation_threshold(self, tf: List[TestFrame], key: int, threshold: float) -> List[List[float]]:
        size = len(tf)
        wm = [[0.0 for _ in range(size)] for _ in range(size)]

        for i in range(size):
            for j in range(size):
                conf_j = tf[j].getFailureProb()
                if conf_j > threshold:
                    wm[i][j] = conf_j
                else:
                    wm[i][j] = 0.0

        return wm

def deepest(testX, testy, dataset, model_name, budget, aux_variable='combo', threshold=0.7):
    dataset_path = f'./AllResult/DeepEST/{model_name}_{dataset}.csv'
    # Determine feature key and adjust threshold
    if aux_variable == "confidence":
        key = 3
        threshold = 1 - threshold
    elif aux_variable == "dsa":
        key = 4
    elif aux_variable == "lsa":
        key = 5
    else:  # combo or others
        key = 6
        threshold = 1 - threshold
    print(f"Approach execution on {dataset_path} with auxiliary variable {aux_variable} and budget {budget}")

    rep = 1 #30
    csv_reader = InitializerTF(dataset_path)
    test_frames = csv_reader.readTestFrames(key, 10000)

    aws = DeepESTSelector()
    weights_matrix = csv_reader.weightedMatrixComputation_threshold(test_frames, key, threshold)
    #rel_arr = []
    #num_fp = []
    #for i in range(rep):
        #rel_arr.append(1 - rel[0])
        #num_fp.append(aws.getnumfp())
    selected_idx = aws.selectAndRunTestCase(budget, test_frames, weights_matrix, 0.8)
    selected_idx = np.array(selected_idx).astype(np.int32)
    return testX[selected_idx], testy[selected_idx], selected_idx
    #for i in range(rep):
    #    print(f"Repetition {i+1}) Estimated Accuracy: {rel_arr[i]:.4f} | Number of failed tests: {num_fp[i]}")

def select(trainX, trainy, X, y, model, budget, metric, dataset, model_name, batch_size=128, **kwargs):
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
        return nac_select(X, y, model, budget, t=kwargs.get('t', 0.75))
    elif metric == 'lsa':
        return lsa_select(trainX, trainy, X, y, model, budget, dataset, model_name, std=kwargs.get('std', 0.05))
    elif metric == 'dsa':
        return dsa_select(trainX, trainy, X, y, model, budget, dataset, model_name, std=kwargs.get('std', 0.05))
    elif metric == 'gd':
        return geometric_diversity_select(X, y, dataset, budget, batch_size, no_groups=kwargs.get('no_groups', 50))
    #elif metric == 'nc':
    #    return neuron_coverage_select(X, y, model, budget, threshold=kwargs.get('threshold', 0.75), batch_size=batch_size)
    elif metric == 'std':
        return std_select(X, y, dataset, budget, batch_size=batch_size, no_groups=50)
    elif metric == 'pace':
        return pace(X, y, model, budget, batch_size=batch_size, layer_idx=kwargs.get('layer_idx', -2), min_cluster_size=kwargs.get('min_cluster_size', 5), min_samples=kwargs.get('min_samples', 5))
    elif metric == 'dr':
        return dr(X, y, model, budget, batch_size=batch_size, KN=kwargs.get('KN', 20))
    elif metric == 'ces':
        return ces_select(X, y, model, budget, batch_size=batch_size, layer_idx=kwargs.get('layer_idx', -3), divide=kwargs.get('divide', 10))
    elif metric == 'mcp':
        return mcp_select(X, y, model, budget, batch_size=batch_size)
    elif metric == 'est':
        return deepest(X, y, dataset, model_name, budget)
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")