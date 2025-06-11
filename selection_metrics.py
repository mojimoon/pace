'''
4 budgets: 50, 100, 150, 200
4 criteria to evaluate selected test suite: APFD, #Clu., RMSE, Acc (%)
16 selection metrics: Rand, Gini, Ent, NC, KMNC, GD, Std, LSA, DSA, CES, PACE, DS, EST, DR, MCP, Dat

This file implements test selection metrics, selecting from each ID (ID.1 to ID.32) testing set

Empericial study references:
https://tjusail.github.io/people/chenjunjie/files/TOSEM20.pdf
https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?params=/context/sis_research/article/8198/&path_info=3511598.pdf

Rand = Random
Gini = DeepGini
Ent = Entropy
NC = Neuron Coverage
KMNC = K-Multisection Neuron Coverage
GD = Geometric Diversity
Std = Standard Deviation
LSA = Likelihood-based Surprise Adequacy
DSA = Distance-based Surprise Adequacy
CES = Cross Entropy-based Sampling (https://github.com/Lizn-zn/DNNOpAcc, https://github.com/Testing-Multiple-DL-Models/SDS)
PACE = Practical Accuracy Estimation (https://github.com/pace2019/pace)
DS = DeepSample (https://github.com/dessertlab/DeepSample)
EST = DeepEST (https://github.com/dessertlab/DeepEST)
DR = DeepReduce (https://github.com/DeepReduce/DeepReduce)
MCP = Multiple-boundary Clustering and Prioritization (https://github.com/actionabletest/MCP)
Dat = Dataset Diversity
'''
import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing

def one_hot_encode(y, num_classes):
    """Convert class vector (integers) to binary class matrix if necessary.
    
    Args:
        y: Shape (n_samples,).
        num_classes: Number of classes.
    Returns:
        Binary class matrix of shape (n_samples, num_classes).
    """
    # return np.eye(num_classes)[y.reshape(-1)]
    if len(y.shape) == 1 or y.shape[1] == 1:
        return np.eye(num_classes)[y.reshape(-1)]
    else:
        return y

def make_batch(X, y, batch_size):
    """Create batches of data without shuffling."""

    n_samples = X.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]

def Random(candidateX, candidatey,budget):
    selection_size = int(budget * candidateX.shape[0])
    select_idx = np.random.choice(np.arange(len(candidateX)), selection_size)
    selected_candidateX, selected_candidatey = candidateX[select_idx], candidatey[select_idx]

    return selected_candidateX, selected_candidatey

def entropy(sess, X, Y, model, budget, batch_size=128):
    with sess.as_default():
        entropies = np.array([0.0])
        for x, y in make_batch(X, Y, batch_size):
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            proba = sess.run(model.y_proba, feed_dict=test_dict)
            proba = np.clip(proba, 1e-8, 1.0)  # Avoid log(0)
            entropy_val = -np.sum(proba * np.log(proba), axis=1)
            entropies = np.hstack((entropies, entropy_val))

    scores = np.array(entropies[1:]).flatten()
    idx = np.argsort(scores)[::-1]  # Largest entropy first
    selected_idx = idx[:int(len(idx) * budget)]
    selectedX = X[selected_idx]
    selectedy = Y[selected_idx]
    return selectedX, selectedy, scores

## deepgini
def deep_metric(pred_test_prob):
    metrics=np.sum(pred_test_prob**2,axis=1)
    rank_lst=np.argsort(metrics)
    return rank_lst

def deepgini(sess, candidateX, candidatey, model, budget, batch_size=128):
    gini = 1 - tf.reduce_sum(model.y_proba ** 2, axis=1)
    with sess.as_default():
        score = np.array([0.0])
        for x, y in make_batch(candidateX, candidatey, batch_size):
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _gini_score = sess.run(gini, feed_dict=test_dict)
            score = np.hstack((score, _gini_score.squeeze()))

    scores = np.array(score[1:]).flatten()
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, scores

def dat(sess, candidateX, candidatey, candidateX_id, candidatey_id, hybrid_test, hybrid_testy, model, id_ratio, budget, batch_size=128):
    select_size = budget * candidateX.shape[0]
    id_select_num = int(select_size * id_ratio)    #e.g., 4 out of 40
    ood_select_num = int(select_size - id_select_num)
    para_there = 0.5 # general default value as specified in the paper.
    # Decide how many ID and OOD data we need
    tot_ood_size = int(candidateX.shape[0] - candidateX_id.shape[0])
    if id_select_num > ood_select_num:
        if id_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)
        else:
            flag_num = id_select_num
            id_select_num = ood_select_num
            ood_select_num = int(flag_num)
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = int(select_size - id_select_num)
        if ood_select_num > tot_ood_size: #this may happen due to small candidate size
            ood_select_num = tot_ood_size
            id_select_num = int(select_size - ood_select_num)

    # print("id num: ", id_select_num)
    # print("ood num: ", ood_select_num)

    # select ID by DeepGini
    _,_, id_scores = deepgini(sess, candidateX_id, candidatey_id, model, budget, batch_size=batch_size)
    # select ratio from the largest scores
    idx = np.argsort(id_scores)[::-1]
    id_selected_idx = idx[:int(id_select_num)]
    selected_candidateX_id = candidateX_id[id_selected_idx]
    selected_candidatey_id = candidatey_id[id_selected_idx]

    # ood data selection
    # top 1 label
    candidate_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: candidateX, model.y_input: candidatey, model.is_training: False})  #np.argmax(model.predict(candidateX), axis=1)
    reference_prediction_label = sess.run(model.y_pred, feed_dict={model.x_input: hybrid_test, model.y_input: hybrid_testy, model.is_training: False}) #np.argmax(model.predict(hybrid_test), axis=1)
    # print(reference_prediction_label)
    reference_labels = []
    for i in range(0, 2):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidateX) / select_size
    reference_labels = reference_labels / s_ratio

    ood_part_index = np.arange(candidateX.shape[0])[candidateX_id.shape[0]:]

    label_list = []
    index_list = []

    for _ in range(1000):
        ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

        this_labels = candidate_prediction_label[ood_select_index.astype(np.int)]
        single_labels = []
        for i in range(0, 2):
            label_num = len(np.where(this_labels == i)[0])
            single_labels.append(label_num)
        index_list.append(ood_select_index)
        label_list.append(single_labels)
    index_list = np.asarray(index_list)
    label_list = np.asarray(label_list)

    # compare to test label
    label_minus = np.abs(label_list - reference_labels)
    var_list = np.sum(label_minus, axis=1)
    var_list_order = np.argsort(var_list)

    ood_select_index = index_list[var_list_order[0]]
    selected_candidateX_ood = candidateX[ood_select_index]
    selected_candidatey_ood = candidatey[ood_select_index]
    selected_data = np.concatenate((selected_candidateX_id, selected_candidateX_ood), axis=0)
    selected_label = np.concatenate((selected_candidatey_id, selected_candidatey_ood), axis=0)

    return selected_data, selected_label # selected candidate data for retraining

## deep gauge
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
        for n_index,neuron_activate in tqdm(enumerate(self.neuron_activate)):
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
            for index in tqdm(lst):
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
            print(max_cover_num)
        return subset

## deepxplore
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
            for index in tqdm(lst):
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
            print(max_cover_num)
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

## LSA
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
                kde = stats.gaussian_kde(temp, bw_method='scott')
                test_mean+=kde.evaluate(temp)
            test_score.append(reduce(lambda x,y:np.log(x)+np.log(y),test_mean/len(self.neuron_activate_train)))
        return test_score

## DSA
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

def select_with_model(sess, candidateX, candidatey, model, budget, selection_metric, **kwargs):
    selection_size = int(budget * candidateX.shape[0])

    if selection_metric == 'kmnc':
        k_bins = kwargs.get('k_bins', 1000)
        s_model = kmnc(candidateX, model.x_input, model.layers, k_bins=k_bins)
        test_score = s_model.fit(candidateX)
        sorted_indices = s_model.rank_fast(candidateX)
        selected_indices = sorted_indices[:selection_size]
        selected_candidateX = candidateX[selected_indices]
        selected_candidatey = candidatey[selected_indices]
        return selected_candidateX, selected_candidatey
    elif selection_metric == 'nac':
        t = kwargs.get('t', 0.5)
        s_model = nac(candidateX, model.x_input, model.layers, t=t)
        test_score = s_model.fit()
        sorted_indices = s_model.rank_fast(candidateX)
        selected_indices = sorted_indices[:selection_size]
        selected_candidateX = candidateX[selected_indices]
        selected_candidatey = candidatey[selected_indices]
        return selected_candidateX, selected_candidatey
    else:
        if selection_metric == 'lsa':
            std = kwargs.get('std', 0.05)
            s_model = LSA(candidateX, model.x_input, model.layers, std=std)
        elif selection_metric == 'dsa':
            std = kwargs.get('std', 0.05)
            label = kwargs.get('label', None)
            if label is None:
                raise ValueError("Label must be provided for DSA selection metric.")
            s_model = DSA(candidateX, label, model.x_input, model.layers, std=std)

        if s_model is None:
            raise ValueError("Invalid selection metric provided: %s" % selection_metric)

        test_score = s_model.fit(candidateX)
        sorted_indices = np.argsort(test_score)[::-1]  # Sort in descending order
        selected_indices = sorted_indices[:selection_size]
        selected_candidateX = candidateX[selected_indices]
        selected_candidatey = candidatey[selected_indices]
        return selected_candidateX, selected_candidatey


def GD(sess, candidateX, candidatey, model, selection_size, number=60, batch_size=128):
    '''
    Black-box diversity test selection metric

    Step 1: Feature Extraction. Extract features for each sample in the candidateX -> Shape (len(canX), m)
    Step 2: Randomly select with replacement N=number groups of size n=budget*len(idx) from the feature matrix.
    Step 3: For each group -> Nomralize column-wise (feature-wise)
    Step 4: Calculate diversity score GD (geometric diversity)
    Step 5: Select the group with the highest GD.
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    # test_input = utils.DataProducer(candidateX, candidatey, batch_size=120, name='test')
    #selection_size = int(budget * candidateX.shape[0])

    if model.model_name.lower() == 'deepdrebin':
        hidden_size = 200
    elif model.model_name.lower() == 'basic_dnn':
        hidden_size = 160

    with sess.as_default():
        feat = np.zeros((1,hidden_size))
        for x, y in make_batch(candidateX, candidatey, batch_size):
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _feat = sess.run(model.dense2, feed_dict=test_dict)
            feat = np.vstack((feat, _feat))
        feat_mat = feat[1:]
        GD_scores, selected_indices = [], []
        for _ in range(number):
            select_idx = np.random.choice(np.arange(len(feat_mat)), selection_size)
            selected_indices.append(select_idx)
            select_group = feat_mat[select_idx] #shape (bg*len, feat)
            # normalize group
            normalize_select_group = min_max_scaler.fit_transform(select_group)
            # compute GD
            GD = np.linalg.det(np.matmul(normalize_select_group, normalize_select_group.T))
            GD_scores.append(GD.squeeze())

        max_idx = np.argmax(np.array(GD_scores))
        chosen_indices = selected_indices[max_idx]
        selected_candidateX, selected_candidatey = candidateX[chosen_indices], candidatey[chosen_indices]

    return selected_candidateX, selected_candidatey

def nc(sess, candidateX, candidatey, model, budget, threshold=0, batch_size=128):
    # coverages = []
    # for x, y in zip(candidateX, candidatey):
    #     activations = []
    #     for l in model.layers:
    #         act = sess.run(l, feed_dict={model.x_input: x, model.is_training: False})
    #         activations.append(act)
    #     flat_activations = np.concatenate([a.flatten() for a in activations])
    #     covered = np.sum(flat_activations > threshold) / len(flat_activations)
    #     coverages.append(covered)
    coverages = np.array([0.0])
    for x, y in make_batch(candidateX, candidatey, batch_size):
        test_dict = {
            model.x_input: x,
            model.y_input: y,
            model.is_training: False
        }
        activations = sess.run(model.activations, feed_dict=test_dict)
        flat_activations = np.concatenate([a.flatten() for a in activations], axis=0)
        covered = np.sum(flat_activations > threshold) / len(flat_activations)
        coverages = np.hstack((coverages, covered))
    coverages = np.array(coverages)
    idx = np.argsort(coverages)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    return candidateX[selected_idx], candidatey[selected_idx], coverages

def std(sess, candidateX, candidatey, model, budget, batch_size=128):
    scores = np.array([0.0])
    for x, y in make_batch(candidateX, candidatey, batch_size):
        test_dict = {
            model.x_input: x,
            model.y_input: y,
            model.is_training: False
        }
        y_proba = sess.run(model.y_proba, feed_dict=test_dict)
        std_score = np.std(y_proba, axis=1)
        scores = np.hstack((scores, std_score))
    scores = np.array(scores[1:]).flatten()
    idx = np.argsort(scores)[::-1] # Largest std first
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, scores

def pace(sess, candidateX, candidatey, model, budget,
         select_layer_tensor,
         min_cluster_size=15, min_samples=10, dec_dim=None, batch_size=128):
    from sklearn.decomposition import FastICA
    import hdbscan
    """
    PACE: Practical Accuracy Estimation
    https://github.com/pace2019/pace/blob/master/driving/selection.py
    select_layer_tensor: 你需要抽取特征的中间层输出（作为聚类输入），如 model.layer_outputs['dense_3']
    min_cluster_size, min_samples, dec_dim: hdbscan/降维参数
    """
    # dense_features = []
    # for x, y in zip(candidateX, candidatey):
    #     test_dict = {
    #         model.x_input: x,
    #         model.y_input: y,
    #         model.is_training: False
    #     }
    #     feature_vec = sess.run(select_layer_tensor, feed_dict=test_dict)
    #     dense_features.append(feature_vec.squeeze())
    # dense_features = np.array(dense_features)
    dense_features = np.zeros((1, model.layer_outputs[select_layer_tensor].shape[-1]))
    for x, y in make_batch(candidateX, candidatey, batch_size):
        test_dict = {
            model.x_input: x,
            model.y_input: y,
            model.is_training: False
        }
        feature_vec = sess.run(model.layer_outputs[select_layer_tensor], feed_dict=test_dict)
        dense_features = np.vstack((dense_features, feature_vec.squeeze()))
    dense_features = dense_features[1:]  # Remove the initial zero row

    dense_features = preprocessing.normalize(dense_features)
    if dec_dim is not None:
        fica = FastICA(n_components=dec_dim)
        dense_features = fica.fit_transform(dense_features)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(dense_features)
    outlier_scores = getattr(clusterer, 'outlier_scores_', np.zeros(len(candidateX)))
    
    pace_scores = outlier_scores
    idx = np.argsort(pace_scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, pace_scores

def dr(sess, candidateX, candidatey, model, budget, num_classes, batch_size=128): # 3 for driving, 10 for mnist
    """
    DR: DeepReduce
    https://github.com/DeepReduce/DeepReduce/blob/master/mnist/lenet-4/DeepReduce-coverage.py
    """

    # candidatey_onehot = one_hot_encode(candidatey, num_classes)

    # kl_scores = np.zeros((1,))
    # for x, y_true in make_batch(candidateX, candidatey_onehot, batch_size):
    #     test_dict = {
    #         model.x_input: x,
    #         model.y_input: y_true,
    #         model.is_training: False
    #     }
    #     y_pred = sess.run(model.y_proba, feed_dict=test_dict).squeeze()
    #     eps = 1e-10
    #     kl = np.sum(y_true * np.log((y_true + eps) / (y_pred + eps)), axis=1)
    #     kl_scores = np.hstack((kl_scores, kl))

    kl_scores = np.zeros((1,))
    for x, y in make_batch(candidateX, candidatey, batch_size):
        y_onehot = one_hot_encode(y, num_classes)
        test_dict = {
            model.x_input: x,
            model.y_input: y,
            model.is_training: False
        }
        y_pred = sess.run(model.y_proba, feed_dict=test_dict).squeeze() # size (batch_size, num_classes)
        eps = 1e-10
        kl = np.sum(y_onehot * np.log((y_onehot + eps) / (y_pred + eps)), axis=1)
        kl_scores = np.hstack((kl_scores, kl))
    
    kl_scores = np.array(kl_scores[1:]).flatten()
    idx = np.argsort(kl_scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, kl_scores

def ces(sess, candidateX, candidatey, model, budget, num_classes, batch_size=128):
    """
    CES: Cross Entropy-based Sampling
    https://github.com/Lizn-zn/DNNOpAcc/blob/master/CE%20method/MNIST/LeNet-4/crossentropy.py
    """

    ce = -tf.reduce_sum(model.y_input * tf.math.log(model.y_proba + 1e-8), axis=1)

    candidatey_onehot = one_hot_encode(candidatey, num_classes)

    scores = np.empty((0,))
    for x, y in make_batch(candidateX, candidatey_onehot, batch_size):
        test_dict = {
            model.x_input: x,
            model.y_input: y,
            model.is_training: False
        }
        ce_score = sess.run(ce, feed_dict=test_dict)
        scores = np.hstack((scores, ce_score))
    scores = np.array(scores[1:]).flatten()
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]
    return selected_candidateX, selected_candidatey, scores


def mcp(sess, candidateX, candidatey, model, budget, num_classes, batch_size=128):
    """
    MCP: Multiple-boundary Clustering and Prioritization
    https://github.com/actionabletest/MCP/blob/master/samedist_mnist_retrain.py
    """

    all_proba = np.empty((0, num_classes))

    # 获取所有样本的 softmax 概率
    for x, y in make_batch(candidateX, candidatey, batch_size):
        test_dict = {
            model.x_input: x,
            model.y_input: y,
            model.is_training: False
        }
        proba = sess.run(model.y_proba, feed_dict=test_dict)
        all_proba = np.vstack((all_proba, proba))

    # MCP打分：最大概率与第二大概率之比
    max_proba = np.max(all_proba, axis=1)
    argsort = np.argsort(all_proba, axis=1)
    second_max_index = argsort[:, -2]
    second_max_proba = all_proba[np.arange(all_proba.shape[0]), second_max_index]
    mcp_scores = second_max_proba / (max_proba + 1e-12)

    scores = mcp_scores
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    selected_candidateX = candidateX[selected_idx]
    selected_candidatey = candidatey[selected_idx]

    return selected_candidateX, selected_candidatey, scores

def deepest(sess, candidateX, candidatey, model, budget, d=0.5, batch_size=128):
    """
    EST: DeepEST
    https://github.com/dessertlab/DeepEST/blob/main/DeepEST_source_code/selector/DeepESTSelector.java
    """
    with sess.as_default():
        all_probs = []
        for x, y in make_batch(candidateX, candidatey, batch_size):
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            p = sess.run(model.y_proba, feed_dict=test_dict)
            all_probs.append(p)
        all_probs = np.vstack(all_probs)   # [N, num_classes]

        occurrenceProb = -np.sum(all_probs * np.log(all_probs + 1e-8), axis=1)  # shape [N,]
        N = len(candidateX)
        n = int(N * budget)
        if n <= 0 or n > N:
            raise ValueError("Invalid budget, check input length and budget ratio.")

        scompl = list(range(N))
        estimationX = np.zeros(n)
        selected_idx = []
        randomNum = np.random.choice(scompl)
        selected_idx.append(randomNum)
        scompl.remove(randomNum)
        y = 1 
        estimationX[0] = N * occurrenceProb[randomNum] * y

        for k in range(1, n):
            if np.random.rand() <= d:
                # d概率下，使用 occurrenceProb 采样
                probs = occurrenceProb[scompl]
                probs = probs / np.sum(probs)
                current_tf = np.random.choice(scompl, p=probs)
            else:
                # (1-d)概率下，均匀随机采样
                current_tf = np.random.choice(scompl)
            selected_idx.append(current_tf)
            scompl.remove(current_tf)
            estimationX[k] = estimationX[k-1] + occurrenceProb[current_tf]

        selected_candidateX = candidateX[selected_idx]
        selected_candidatey = candidatey[selected_idx]
        scores = occurrenceProb
        return selected_candidateX, selected_candidatey, scores

def select(sess, candidateX, candidatey, model, budget, selection_metric, **kwargs):
    if selection_metric == 'random':
        return Random(candidateX, candidatey, budget)
    elif selection_metric == 'entropy':
        return entropy(sess, candidateX, candidatey, model, budget)
    elif selection_metric == 'deepgini':
        return deepgini(sess, candidateX, candidatey, model, budget)
    elif selection_metric == 'dat':
        candidateX_id = kwargs.get('candidateX_id', None)
        candidatey_id = kwargs.get('candidatey_id', None)
        hybrid_test = kwargs.get('hybrid_test', None)
        hybrid_testy = kwargs.get('hybrid_testy', None)
        id_ratio = kwargs.get('id_ratio', 0.5)
        return dat(sess, candidateX, candidatey, candidateX_id, candidatey_id, hybrid_test, hybrid_testy, model, id_ratio, budget)
    elif selection_metric == 'gd':
        return GD(sess, candidateX, candidatey, model, budget)
    elif selection_metric in ['kmnc', 'nac', 'lsa', 'dsa']:
        return select_with_model(sess, candidateX, candidatey, model, budget, selection_metric, **kwargs)
    elif selection_metric == 'nc':
        threshold = kwargs.get('threshold', 0)
        return nc(sess, candidateX, candidatey, model, budget, threshold)
    elif selection_metric == 'std':
        return std(sess, candidateX, candidatey, model, budget)
    elif selection_metric == 'pace':
        select_layer_tensor = kwargs.get('select_layer_tensor', 'dense_3')
        min_cluster_size = kwargs.get('min_cluster_size', 15)
        min_samples = kwargs.get('min_samples', 10)
        dec_dim = kwargs.get('dec_dim', None)
        return pace(sess, candidateX, candidatey, model, budget, select_layer_tensor, min_cluster_size, min_samples, dec_dim)
    elif selection_metric == 'dr':
        num_classes = kwargs.get('num_classes', 10)
        return dr(sess, candidateX, candidatey, model, budget, num_classes)
    elif selection_metric == 'ces':
        num_classes = kwargs.get('num_classes', 10)
        return ces(sess, candidateX, candidatey, model, budget, num_classes)
    elif selection_metric == 'mcp':
        num_classes = kwargs.get('num_classes', 10)
        return mcp(sess, candidateX, candidatey, model, budget, num_classes)
    elif selection_metric == 'est':
        d = kwargs.get('d', 0.5)
        return deepest(sess, candidateX, candidatey, model, budget, d)


# if __name__ == '__main__':
#     (_, _), (X_test, Y_test) = mnist.load_data()
#     X_test = X_test.astype('float32')
#     X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#     X_test /= 255
#     y_test = to_categorical(Y_test, 10)
#     budget = 50
#     my_model = keras.models.load_model(os.path.join(basedir, exp_model_dict[exp_id]))
#     x,y,score = entropy(X_test, y_test, model, budget)

