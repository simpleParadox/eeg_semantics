import sklearn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gensim
from numpy import savez_compressed
from numpy import load
import platform
import time
import random
import os
from copy import deepcopy
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gensim
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
# from sklearn.svm import LinearSVR
# from sklearn.svm import SVR
# from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


labels_mapping = {0:'baby', 1:'bear', 2:'bird', 3: 'bunny',
                      4:'cat', 5 : 'dog', 6: 'duck', 7: 'mom',
                      8: 'banana', 9: 'bottle', 10: 'cookie',
                      11: 'cracker', 12: 'cup', 13: 'juice',
                      14: 'milk', 15: 'spoon'}

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

os_name = platform.system()

if os_name == 'Windows':
    from regression.functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, test_model, test_model_permute, two_vs_two_test, divide_by_labels, random_subgroup, average_grouped_data, get_w2v_embeds
else:
    from functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, test_model, test_model_permute, two_vs_two_test, divide_by_labels, random_subgroup, average_grouped_data, get_w2v_embeds

readys_path = None
avg_readys_path = None
if os_name =='Windows':
    readys_path = "G:\\jw_lab\\jwlab_eeg\\regression\\data\\ml_df_readys.pkl"
    avg_readys_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_data_readys.pkl"
    avg_trials_and_ps_9m_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_9m.pkl"
    avg_trials_and_ps_13m_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_13m.pkl"
    avg_trials_and_ps_9and13_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_9and13.pkl"
    bag_of_features = "G:\\jw_lab\\jwlab_eeg\\regression\data\\bagOfFeatures (1).mat"
elif os_name=='Linux':
    readys_path = os.getcwd() + "/regression/data/ml_df_readys.pkl"
    avg_readys_path = os.getcwd() + "/regression/data/avg_trials_data_readys.pkl"
    avg_trials_and_ps_9m_path = os.getcwd() + "/regression/data/avg_trials_and_ps_9m.pkl"
    avg_trials_and_ps_13m_path = os.getcwd() + "/regression/data/avg_trials_and_ps_13m.pkl"
    avg_trials_and_ps_9and13_path = os.getcwd() + "/regression/data/avg_trials_and_ps_9and13.pkl"
    bag_of_features = os.getcwd() + "/regression/data/bagOfFeatures (1).mat"

# with open(pkl_path, 'rb') as f:
# f = open(readys_path, 'rb')
# readys_data = pickle.load(f)
# f.close()





# eeg_features = readys_data.iloc[:,:18000].values
w2v_path = None
avg_w2v_path = None
if os_name =='Windows':
    w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds.npz"
    avg_w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds_avg_trial.npz"
    gen_w2v_all_ps_avg_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\gen_w2v_embeds_avg_trial_and_ps.npz"
    embeds_with_label_path = 'G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\embeds_with_label_dict.npz'
    bof_embeds_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\bof_w2v_embeds.npz"
elif os_name=='Linux':
    w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds.npz"
    avg_w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds_avg_trial.npz"
    gen_w2v_all_ps_avg_path = os.getcwd() + "/regression/w2v_embeds/gen_w2v_embeds_avg_trial_and_ps.npz"
    embeds_with_label_path = os.getcwd() + "/regression/w2v_embeds/embeds_with_label_dict.npz"
    bof_embeds_path = os.getcwd() + "/regression/w2v_embeds/bof_w2v_embeds.npz"




def monte_carlo_2v2_permuted(X, Y, split_idxs):
    # start = time.time()
    # random.shuffle(Y)
    print("Monte-Carlo CV DT Permuted")
    # Split into training and testing data
    parameters_ridge = {'alpha': [0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'min_samples_split': [4, 6, 8, 10, 20]}  #



    dt = Ridge()
    clf = GridSearchCV(dt, param_grid=parameters_ridge, scoring='neg_mean_squared_error',
                       refit=True, cv=5, n_jobs=1)

    eeg_features = X# readys_data.iloc[:, :].values  # :208 for thirteen month olds. 208: for nine month olds.
    w2v_embeds_mod = Y# w2v_embeds[:]  # :208 for thirteen month olds. 208: for nine month olds.

    # print(eeg_features.shape)
    # print(w2v_embeds_mod.shape)
    # rs = ShuffleSplit(n_splits=10, train_size=0.90)
    all_data_indices = [i for i in range(len(w2v_embeds_mod))]
    f = 1
    score_with_alpha = {}
    cosine_scores = []
    # for train_index, test_index in rs.split(all_data_indices):
    for idxs in split_idxs:
        print("Shuffle Split fold: ", f)
        train_idx = idxs[0]
        test_idx = idxs[1]
        # print("train index: ", train_idx)
        # print("test index: ", test_idx)
        X_train, X_test = eeg_features[train_idx], eeg_features[test_idx]
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        # print("Train index before", train_index)
        random.shuffle(train_idx) # For permutation test only.
        random.shuffle(test_idx) # For permutation test only.
        # print("Train index after: ", train_index)
        y_train, y_test = w2v_embeds_mod[train_idx], w2v_embeds_mod[test_idx]

        # ss = StandardScaler()
        # X_train = ss.fit_transform(X_train)
        # X_test = ss.transform(X_test)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        # print("Preds", preds.shape)
        # print("y_test:", y_test.shape)
        f += 1
        points, total_points, score = two_vs_two(y_test, preds)
        # print("Points: ", points)
        # print("Total points: ", total_points)
        acc = points / total_points
        cosine_scores.append(acc)
        # print(acc)
    score_with_alpha['avg'] = np.average(np.array(cosine_scores), axis=0)
    # print("All scores: ", score_with_alpha)
    # stop = time.time()
    # print("Total time: ", stop - start)
    return score_with_alpha['avg']


def monte_carlo_2v2(X,Y):
    # start = time.time()
    print("Monte-Carlo CV DT Normal")
    # Split into training and testing data
    parameters_ridge = {'alpha': [0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'min_samples_split': [4, 6, 8, 10, 20]}  #



    dt = Ridge()
    clf = GridSearchCV(dt, param_grid=parameters_ridge, scoring='neg_mean_squared_error',
                       refit=True, cv=5, n_jobs=1)

    eeg_features = X# readys_data.iloc[:, :].values  # :208 for thirteen month olds. 208: for nine month olds.
    w2v_embeds_mod = Y# w2v_embeds[:]  # :208 for thirteen month olds. 208: for nine month olds.

    # print(eeg_features.shape)
    # print(w2v_embeds_mod.shape)
    rs = ShuffleSplit(n_splits=5, train_size=0.90)
    all_data_indices = [i for i in range(len(w2v_embeds_mod))]
    f = 1
    score_with_alpha = {}
    cosine_scores = []
    shuffle_split_idxs = []
    for train_index, test_index in rs.split(all_data_indices):
        print("Shuffle Split fold: ", f)
        shuffle_split_idxs.append([train_index, test_index])
        X_train, X_test = eeg_features[train_index], eeg_features[test_index]
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        # print("Train index before", train_index)
        # random.shuffle(train_index) # For permutation test only.
        # random.shuffle(test_index) # For permutation test only.
        # print("Train index after: ", train_index)
        y_train, y_test = w2v_embeds_mod[train_index], w2v_embeds_mod[test_index]

        # ss = StandardScaler()
        # X_train = ss.fit_transform(X_train)
        # X_test = ss.transform(X_test)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        # print("Preds", preds.shape)
        # print("y_test:", y_test.shape)
        f += 1
        points, total_points, score = two_vs_two(y_test, preds)
        # print("Points: ", points)
        # print("Total points: ", total_points)
        acc = points / total_points
        cosine_scores.append(acc)
        # print(acc)
    score_with_alpha['avg'] = np.average(np.array(cosine_scores), axis=0)
    # print("All scores: ", score_with_alpha)
    # stop = time.time()
    # print("Total time: ", stop - start)
    return score_with_alpha['avg'], shuffle_split_idxs



"""
Run some test experiments.
    1. Load the EEG data.
    2. Create the Word2Vec embeddings.
    3. Align the word length from the w2v embeddings to the EEG data.
    4. Fit model.
    5. Predict.
"""
# 1. Load the EEG Data.
f = open(readys_path, 'rb')
readys_data = pickle.load(f)
f.close()

# 2. Load the BoF Data.
bof_data = loadmat(bag_of_features)

# Align data.
# def align_data():
# First extract the labels.
labels = readys_data.iloc[:, 18000].values
eeg_data = readys_data.iloc[:, :18000].values
# Now create a new list with the length of the words in it.
word_labels = []
model = gensim.models.KeyedVectors.load_word2vec_format('G:\jw_lab\jwlab_eeg\\regression\GoogleNews-vectors-negative300.bin.gz', binary=True)
for label in labels:
    word_labels.append(model[labels_mapping[int(label)]])

correct_labels = deepcopy(np.array(word_labels))

# tsne = TSNE(n_components=100, perplexity=50,method='exact')
# eeg_data = tsne.fit_transform(eeg_data, correct_labels)


word_labels = np.array(word_labels)

correct_scores = []
for i in range(0, 100):
    correct_scores.append(test_model(eeg_data, correct_labels, deepcopy(labels)))
print("Correct Scores Average: ", np.mean(correct_scores))

permuted_scores = []
idxs = [_ for _ in range(len(word_labels))]
for i in range(0, 100):
    random.shuffle(idxs)
    word_labels = word_labels[idxs]
    permuted_scores.append(test_model_permute(eeg_data, deepcopy(word_labels), deepcopy(labels)))

print("Permuted Scores Average: ", np.mean(permuted_scores))


# # Plot the variance vs components graph
# pca = PCA(n_components=4)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(eeg_data)
# X_mod = pca.fit_transform(X_scaled)
# comps = pca.components_
# variances = pca.explained_variance_
# var_list = [_ for _ in range(len(variances))]
# print(comps.shape)
# print(variances.shape)
# var_ratio = pca.explained_variance_ratio_
# plt.plot(var_list, var_ratio)
# plt.xlim([0,10])
# plt.show()

## First label the words as animate or inanimate.
## Predic the word2vec embeddings
# How about train on one group of labels and predict on others?
# This should be based on the similarity of the words. Let's try animates and inanimates only.
