import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
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
from sklearn.decomposition import TruncatedSVD, PCA
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


labels_mapping_mod_ratings = {0:'baby', 1:'bear', 2:'bird', 3: 'rabbit',
                      4:'cat', 5 : 'dog', 6: 'duck',
                      8: 'banana', 9: 'bottle', 10: 'cookie',
                      11: 'biscuit', 12: 'cup', 13: 'juice',
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
    # readys_path = "Z:\\Jenn\\ml_df_readys.pkl"
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

# bof_data = loadmat(bag_of_features)



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


def get_w2v_embeds(labels):

    model = gensim.models.KeyedVectors.load_word2vec_format('G:\jw_lab\jwlab_eeg\\regression\GoogleNews-vectors-negative300.bin.gz', binary=True)
    all_embeds = []
    present_indices = []
    idx = 0
    for i in labels:
        word = i[0][0].lower()
        try:
            all_embeds.append(model[word])
            present_indices.append(idx)
        except:
            print("Not in vocabulary")
            print(idx)
        idx += 1
    print("Total length: ", len(all_embeds))
    savez_compressed('/regression/w2v_embeds/bof_w2v_embeds.npz', all_embeds)
    # for label in labels:
    #     all_embeds.append(w2v_label_embeds[int(label)])
    # return all_embeds


def monte_carlo_2v2_permuted(X, Y, split_idxs=None):
    # start = time.time()
    # random.shuffle(Y)
    print("Monte-Carlo CV DT Permuted")
    # Split into training and testing data
    # parameters_ridge = {'alpha': [0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_rf = {'n_estimators': [100, 150]}  # , 'min_samples_split': [2]}#, 5, 10], }  #

    dt = RandomForestRegressor(n_jobs=24)
    clf = GridSearchCV(dt, param_grid=parameters_rf, scoring='neg_mean_squared_error',
                       refit=True, cv=5, n_jobs=1, verbose=2)

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
        # print("Shuffle Split fold: ", f)
        train_idx = idxs[0]
        test_idx = idxs[1]
        # print("train index: ", train_idx)
        # print("test index: ", test_idx)
        X_train, X_test = eeg_features[train_idx], eeg_features[test_idx]
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        # print("Train index before", train_index)
        random.shuffle(train_idx) # For permutation test only.
        random.shuffle(test_idx) # For permutation test only.
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
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


def monte_carlo_2v2(X,Y, permuted=False):
    # start = time.time()
    print("Monte-Carlo CV DT Normal")
    # Split into training and testing data
    # parameters_ridge = {'alpha': [0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_rf = {'n_estimators': [100, 150]}#, 'min_samples_split': [2]}#, 5, 10], }  #



    dt = RandomForestRegressor(n_jobs=10)
    clf = GridSearchCV(dt, param_grid=parameters_rf, scoring='neg_mean_squared_error',
                       refit=True, cv=5, n_jobs=3, verbose=2)

    eeg_features = X# readys_data.iloc[:, :].values  # :208 for thirteen month olds. 208: for nine month olds.
    w2v_embeds_mod = Y# w2v_embeds[:]  # :208 for thirteen month olds. 208: for nine month olds.

    # print(eeg_features.shape)
    # print(w2v_embeds_mod.shape)
    rs = ShuffleSplit(n_splits=3, train_size=0.80)
    all_data_indices = [i for i in range(len(w2v_embeds_mod))]
    f = 1
    score_with_alpha = {}
    cosine_scores = []
    shuffle_split_idxs = []
    for train_index, test_index in rs.split(all_data_indices):
        # print("Shuffle Split fold: ", f)
        shuffle_split_idxs.append([train_index, test_index])
        X_train, X_test = eeg_features[train_index], eeg_features[test_index]
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        # print("Train index before", train_index)
        if permuted==True:
            # print("Permuted")
            random.shuffle(train_index) # For permutation test only.
            random.shuffle(test_index) # For permutation test only.
            np.random.shuffle(train_index)
            np.random.shuffle(test_index)
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



# a = bof_data['features'][:, :218]
# b_loaded = load(bof_embeds_path)
# b = b_loaded['arr_0']
#
# nouns = [temp[0][0].lower() for temp in bof_data['nouns']]


# def select_ratings():
# print("Select ratings")
f = open(readys_path, 'rb')  # Load readys_data.
readys_data = pickle.load(f)
f.close()
# Store, indices from the ratings dataset, labels from the readys_data.
# First obtain the indices of the ratings for the words present in the labels_mapping_mod_ratings
# rating_idxs = []
# for w in labels_mapping_mod_ratings:
#     word = labels_mapping_mod_ratings[w]
#     for n_idx in range(len(nouns)):
#         if word == nouns[n_idx]:
#             rating_idxs.append(n_idx)
#             break
# print(rating_idxs)
# print(len(rating_idxs))


#  Create mapping
# labels = readys_data.iloc[:, 18000].values
# # First note the indices where the the label is 7 -> word 'mom
# filtered_idxs = []  # Contains indices from readys_data without the label '7'.
# for l in range(len(labels)):
#     if labels[l] != 7:
#         filtered_idxs.append(l)
# filtered_readys_data = readys_data.iloc[filtered_idxs,:18000].values # Store this
# readys_ratings = []  # Store this. The eeg data and corresponding ratings.
# for j in filtered_idxs:
#     lab_idx = labels[j]  # This is in agreement to the labels_mod_mapping.
#     # print(lab_idx)
#     word = labels_mapping_mod_ratings[lab_idx]
#     bof_rating_index = nouns.index(word)
#     rating = a[bof_rating_index]
#     readys_ratings.append(rating)
# readys_ratings = np.array(readys_ratings)
# The EEG data and ratings are aligned now with the label '7' removed.


# tsvd = TruncatedSVD(n_components=50)
# readys_ratings = tsvd.fit_transform(readys_ratings)

# readys_tsvd = TruncatedSVD(n_components=300)
# filtered_readys_data = readys_tsvd.fit_transform(filtered_readys_data)
#
# test_model(filtered_readys_data, deepcopy(readys_ratings))
# test_model_permute(filtered_readys_data, deepcopy(readys_ratings))







def simple_mod_cv():
    wo_score, shfl_idxs = monte_carlo_2v2(filtered_readys_data, deepcopy(readys_ratings))
    print(wo_score)
    wi_score = monte_carlo_2v2_permuted(filtered_readys_data, deepcopy(readys_ratings), shfl_idxs)
    print(wi_score)


# simple_mod_cv()

def simple_cv():
    idxs = [i for i in range(0, 1000) if i not in [162, 473, 925]]
    wo_score, shfl_idxs = monte_carlo_2v2(a[idxs], deepcopy(b))
    print(wo_score)
    wi_score = monte_carlo_2v2_permuted(a[idxs], deepcopy(b), shfl_idxs)
    print(wi_score)

# simple_cv()

#  Scratches and rough code

# In the following section I find the first k columns that have the lowest average standard deviation across all labels.
# Then use those columns from the EEG data to predict ratings and also the embeddings directly using two separate experiments.


# readys_data_9m = readys_data[1008:]
# # readys_data_9m = readys_data[:1008]
# col_idxs = [t for t in range(18000)]
# std_list = []
# for lab in range(0, 16):
#     inspect = readys_data[readys_data['label'] == lab]
#     std_list.append(np.std(inspect.iloc[:,:18000].values, axis=0))
# std_list = np.average(std_list, axis=0).tolist()
#
# # Now find the first 300 columns that have the minimum standard deviation
# list1, list_idxs = zip(*sorted(zip(std_list, col_idxs)))
#
# all_w2v_embeds_loaded = load(w2v_path)
# all_w2v_embeds = all_w2v_embeds_loaded['arr_0']
#
# w2v_embeds = all_w2v_embeds[1008:]
# w2v_idxs = [t for t in range(len(w2v_embeds))]
# scores = []
# for r in range(20):
#     # random.shuffle(w2v_idxs)
#     # w2v_embeds = w2v_embeds[w2v_idxs]
#     scores.append(test_model(readys_data_9m.iloc[:, list(list_idxs)[:500]].values, deepcopy(w2v_embeds)))
# print("Average", np.average(scores))



# Step 1: Filter the data based on animate and inanimate words

animate_words = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]

inanimate_words = [8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]


def binary_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


def filter_data_by_words(df, type):
    """
    Filter words by animate or inanimate.
    """
    reduced_data = None
    if type == 'animate':
        reduced_data = df[df['label'].isin(animate_words)]
    elif type == 'inanimate':
        reduced_data = df[df['label'].isin(inanimate_words)]
    else:
        reduced_data = df
    return reduced_data

# Step 2: Assign labels to the correct word embeddings and then
# def train():
type = 'all'
df = filter_data_by_words(deepcopy(readys_data), type)  # Contains only animate or inanimate words.

embeds_with_labels_dict_loaded = load(embeds_with_label_path, allow_pickle=True)
embeds_with_labels_dict = embeds_with_labels_dict_loaded['arr_0']
embeds_with_labels_dict = embeds_with_labels_dict[0]
label_numbers = df.iloc[:, 18000].values

# Do PCA on data.
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df.iloc[:, :18000].values)
# pca = PCA(n_components=50)
# X = pca.fit_transform(X_scaled)

# X_classify = df.iloc[:, :].values

# word_types = []
# for label in label_numbers:
#     if label in animate_words:
#         word_types.append(0)
#     else:
#         word_types.append(1)


X = df.iloc[:, :18000].values

# Binary classification correct labels
# binary_classification(deepcopy(X_scaled), word_types)
#
#
# # Binary classification permuted labels
# word_types_permuted = deepcopy(word_types)
# np.random.shuffle(word_types_permuted)
# binary_classification(deepcopy(X), word_types_permuted)




# tsne = TSNE(n_components=3)
# x_pca = tsne.fit_transform(X)
#
#
# colors = label_numbers.astype(int).tolist()
# # handles = [lp(i) for i in np.unique(colors)]
#
# legends = label_numbers.astype(int).astype(str).tolist()
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection='3d')
# ax.margins(0.5,1,1)
# for c in range(len(x_pca)):
#     ax.scatter(x_pca[c, 0], x_pca[c, 1], x_pca[c, 2], s=0.5, label=colors[c])
#
#
# plt.title("T-SNE on EEG all words all subjects")
#
# plt.show()

# var_ratios = pca.explained_variance_ratio_
# cum_var_ratios = np.cumsum(var_ratios)
# comps = [_ for _ in range(len(cum_var_ratios))]
# plt.plot(comps, cum_var_ratios)
# plt.xlabel("Components")
# plt.ylabel("Variance explained (ratio)")
# plt.title("Ratio of total variance explained by components animate")
# plt.show()

w2v_labels = []
for label in label_numbers:
    w2v_labels.append(embeds_with_labels_dict[int(label)])
#
#
w2v_labels = np.array(w2v_labels)

# np.random.shuffle(X)
score = test_model(X, deepcopy(w2v_labels))
print(score)

# print("RandomForest Monte-Carlo CV score all features 8/5 CV, hyper-paramaters = n_estimators")
# score, shuffle_idxs = monte_carlo_2v2(deepcopy(X), deepcopy(w2v_labels))
# print("Monte Carlo RandomForest score",score)
#
#
#
# print("Testing 2 shuffle splits RandomForest Permuted Monte-Carlo CV score all features 8/5 CV, hyper-parameters = n_estimators")
# y = deepcopy(w2v_labels)
# np.random.shuffle(y)
# permute_score = monte_carlo_2v2_permuted(deepcopy(X), deepcopy(y), shuffle_idxs)  # Permuting the labels and calling the same function.
# print(permute_score)

#
#
# # Check if the labels are present in the w2v_labels. -> Result => They are.
# # true_list = []
# # for val in embeds_with_labels_dict.values():
# #     for _ in range(len(w2v_labels)):
# #         if np.array_equal(val, w2v_labels[_]):
# #             true_list.append(True)
#
#
#
# # w2v_scaler = StandardScaler()
# # w2v_labels = w2v_scaler.fit_transform(w2v_labels)
#
# scores = []
# for i in range(1000):
# score = test_model(deepcopy(X), deepcopy(w2v_labels), embeds_with_labels_dict)
#     # scores.append(score)
# print(score)
# # print(np.mean(scores))

# # Permuting the labels here.
# y = deepcopy(w2v_labels)
# permute_scores = []
# for j in range(1000):
#     np.random.shuffle(y)
#     permute_score = test_model(deepcopy(X), y)
#     permute_scores.append(permute_score)
# print(permute_score)

# print(np.mean(permute_scores))

# y = deepcopy(w2v_labels)
# #
# np.random.shuffle(y)
# permute_score = test_model(deepcopy(X), y)
# print(permute_score)
#
#
# # # y_indices = [i for i in range(len(w2v_labels))]
# # # random.shuffle(y_indices)
# # # y = w2v_labels[y_indices]
#
# # train()

