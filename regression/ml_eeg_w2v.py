import sklearn
import pandas as pd
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
from sklearn.preprocessing import PolynomialFeatures
import gensim
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
# from sklearn.svm import LinearSVR
# from sklearn.svm import SVR
# from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

os_name = platform.system()

if os_name == 'Windows':
    from regression.functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, test_model, test_model_permute, two_vs_two_test, divide_by_labels, random_subgroup, average_grouped_data, get_w2v_embeds
else:
    from functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, two_vs_two_test, test_model, test_model_permute, divide_by_labels, random_subgroup, average_grouped_data, get_w2v_embeds






readys_path = None
avg_readys_path = None
if os_name =='Windows':
    readys_path = "G:\\jw_lab\\jwlab_eeg\\regression\\data\\ml_df_readys.pkl"
    avg_readys_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_data_readys.pkl"
    avg_trials_and_ps_9m_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_9m.pkl"
    avg_trials_and_ps_13m_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_13m.pkl"
    avg_trials_and_ps_9and13_path = "G:\\jw_lab\\jwlab_eeg\\regression\data\\avg_trials_and_ps_9and13.pkl"
    bag_of_features = "G:\jw_lab\jwlab_eeg\Scratches\data\\bagOfFeatures (1).mat"
elif os_name=='Linux':
    readys_path = os.getcwd() + "/regression/data/ml_df_readys.pkl"
    avg_readys_path = os.getcwd() + "/regression/data/avg_trials_data_readys.pkl"
    avg_trials_and_ps_9m_path = os.getcwd() +  "/regression/data/avg_trials_and_ps_9m.pkl"
    avg_trials_and_ps_13m_path = os.getcwd() + "/regression/data/avg_trials_and_ps_13m.pkl"
    avg_trials_and_ps_9and13_path = os.getcwd() + "/regression/data/avg_trials_and_ps_9and13.pkl"

# with open(pkl_path, 'rb') as f:
# f = open(readys_path, 'rb')
# readys_data = pickle.load(f)
# f.close()

# bof_data = loadmat('/regression/data/bagOfFeatures (1).mat')




# eeg_features = readys_data.iloc[:,:18000].values
w2v_path = None
avg_w2v_path = None
if os_name =='Windows':
    w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds.npz"
    avg_w2v_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\all_w2v_embeds_avg_trial.npz"
    gen_w2v_all_ps_avg_path = "G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\gen_w2v_embeds_avg_trial_and_ps.npz"
    embeds_with_label_path = 'G:\\jw_lab\\jwlab_eeg\\regression\\w2v_embeds\\embeds_with_label_dict.npz'
elif os_name=='Linux':
    w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds.npz"
    avg_w2v_path = os.getcwd() + "/regression/w2v_embeds/all_w2v_embeds_avg_trial.npz"
    gen_w2v_all_ps_avg_path = os.getcwd() + "/regression/w2v_embeds/gen_w2v_embeds_avg_trial_and_ps.npz"
    embeds_with_label_path = os.getcwd() + "/regression/w2v_embeds/embeds_with_label_dict.npz"
# w2v_embeds_loaded = load(gen_w2v_all_ps_avg_path)
# w2v_embeds = w2v_embeds_loaded['arr_0']

# print("Data Loaded")
# print("Readys shape: ", readys_data.shape)
# print("w2v shape: ", w2v_embeds.shape)
# print("data type: ", type(readys_data))
# print("w2v type", type(w2v_embeds))


  # The last value is the score.






def get_ps():
    # participants = ["904", "905", "906", "909", "910", "912", "908", "913", "914", "916", "917", "919", "920", "921", "923", "924","927", "928", "929", "930", "932"]
    # participants = [ "909", "910", "912", "908", "913", "914", "916", "917", "919", "920", "921", "923", "924","927", "928", "929", "930", "932"]
    # 9m with >40 trials
    # participants = [ "909", "912", "908", "913", "914", "916", "917", "919", "920", "921", "924","927", "930"]

    # 12m all
    # participants = ["105", "107", "109", "111", "112", "115", "116", "117", "119", "121", "122", "120", "124"]
    # 12m with >40 trials
    # participants = ["109", "111", "112", "115", "124"]

    # all participants
    participants = [ "904", "905", "906", "909", "910", "912", "908", "913", "914", "916", "917", "919", "920", "921", "923", "924","927", "928", "929", "930", "932",
                  "105", "107", "109", "111", "112", "115", "116", "117", "119", "121", "122", "120", "124"]
    return participants


def create_avg_data_embeds():
    # First average the trials
    ps = get_ps()
    avg_trial_ps_data, labels, participants, labels_copy = average_trials_and_participants(readys_data, ps)

    all_avg_df = pd.DataFrame(avg_trial_ps_data)
    all_avg_df['label'] = labels
    all_avg_df['participants'] = participants
    # Now create the word2vec embeddings from the labels.
    # word_labels = readys_data['label'].values
    # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    # # First obtain all the embeddings for the words in the labels mapping.
    # w2v_label_embeds = {}
    # # pca = PCA(n_components=15)
    # for key in labels_mapping:
    #     w2v_label_embeds[key] = model[labels_mapping[key]]
    # all_embeds = []
    # for label in labels:
    #     all_embeds.append(w2v_label_embeds[int(label)])
    # savez_compressed('w2v_embeds/gen_w2v_embeds_avg_trial_and_ps_test.npz', all_embeds)

    # avg_data = pd.DataFrame(avg_trial_ps_data)
    # avg_data.to_pickle('data/avg_trials_and_ps_9and13.pkl')

# create_avg_data_embeds()

def split_ps_model():
    start = time.time()
    # Split the readys data into 13 month and 9 month olds.
    # last 13 month old index => 1007.
    t_eeg = eeg_features[:1008, :]
    n_eeg = eeg_features[1008:, :]



    t_w2v = w2v_embeds[:1008, :]
    n_w2v = w2v_embeds[1008:, :]
    # Fitting on 9 month old.
    rounds = 100
    t_scores = []
    n_scores = []
    for r in range(rounds):

        t_X_train, t_X_test, t_y_train, t_y_test = train_test_split(t_eeg, t_w2v, train_size=0.90)
        n_X_train, n_X_test, n_y_train, n_y_test = train_test_split(n_eeg, n_w2v, train_size=0.90)

        print("Fitting on the 13 month olds.")
        t_model = DecisionTreeRegressor()
        t_model.fit(t_X_train, t_y_train)
        t_preds = t_model.predict(t_X_test)
        t_points, t_total_points, t_score = two_vs_two(t_y_test, t_preds)
        t_scores.append(t_score)
        print("Score for 13 month olds - no cv, no hyper optim: ", t_score)
        print("------------------------------------------------------------")
        print("Fitting on the 9 month olds.")
        n_model = DecisionTreeRegressor()
        n_model.fit(n_X_train, n_y_train)
        n_preds = n_model.predict(n_X_test)
        n_points, n_total_points, n_score = two_vs_two(n_y_test, n_preds)
        n_scores.append(n_score)
        print("Score for 9 month olds - no cv, no hyper optim: ", n_score)
    print("Average score for thirteen month old: ", np.average(t_scores))
    print("Average score for nice month old: ", np.average(n_scores))
    stop = time.time()
    print("Total time taken: ", stop - start)


def do_svd(ref):
    tsvd = TruncatedSVD(n_components=500)
    ref = tsvd.fit_transform(ref)
    return ref


def monte_carlo_2v2_permuted(X, Y, split_idxs):
    # start = time.time()
    # NOTE: Shuffling of data should be done before calling this function.
    print("Monte-Carlo CV DT Permuted")
    # Split into training and testing data
    parameters_ridge = {'alpha': [10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'min_samples_split': [4, 6, 8, 10, 20]}  #



    dt = Ridge()
    clf = GridSearchCV(dt, param_grid=parameters_dt, scoring='neg_mean_squared_error',
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
        # print("Shuffle Split fold: ", f)
        train_idx = idxs[0]
        test_idx = idxs[1]
        # print("train index: ", train_idx)
        # print("test index: ", test_idx)
        X_train, X_test = eeg_features.iloc[train_idx, :].values, eeg_features.iloc[test_idx,:].values
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        # print("Train index before", train_index)
        # random.shuffle(train_idx) # For permutation test only.
        # random.shuffle(test_idx) # For permutation test only.
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


def monte_carlo_2v2_modified(X, Y):  # Train on correct, test on permuted. Use the same model to test on permuted.
    # start = time.time()
    print("Monte-Carlo CV DT Modified")
    # Split into training and testing data
    parameters_ridge = {'alpha': [10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'min_samples_split': [4, 6, 8, 10, 20]}  #



    dt = Ridge()
    clf = GridSearchCV(dt, param_grid=parameters_dt, scoring='neg_mean_squared_error',
                       refit=True, cv=5, n_jobs=4)

    eeg_features = X# readys_data.iloc[:, :].values  # :208 for thirteen month olds. 208: for nine month olds.
    w2v_embeds_mod = Y# w2v_embeds[:]  # :208 for thirteen month olds. 208: for nine month olds.

    # print(eeg_features.shape)
    # print(w2v_embeds_mod.shape)
    rs = ShuffleSplit(n_splits=10, train_size=0.90)
    all_data_indices = [i for i in range(len(w2v_embeds_mod))]
    f = 1
    score_with_alpha = {}
    score_with_alpha_modified = {}
    cosine_scores = []
    cosine_scores_modified = []
    shuffle_split_idxs = []
    for train_index, test_index in rs.split(all_data_indices):
        # print("Shuffle Split fold: ", f)
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
        points, total_points, score = two_vs_two(y_test, preds)
        cosine_scores.append(score)  # Correct labels

        # Permute the test labels here.
        mod_permute_indices = [_ for _ in range(len(y_test))]
        random.shuffle(mod_permute_indices)
        y_test = y_test[mod_permute_indices]

        # preds = clf.predict(X_test)
        # print("Preds", preds.shape)
        # print("y_test:", y_test.shape)

        points, total_points, score = two_vs_two(y_test, preds)
        cosine_scores_modified.append(score)
        f += 1
        # print(acc)
    score_with_alpha['avg'] = np.average(np.array(cosine_scores), axis=0)
    score_with_alpha_modified['avg'] = np.average(np.array(cosine_scores_modified), axis=0)
    # print("All scores correct: ", score_with_alpha)
    # print("All scores permuted modified: ", score_with_alpha_modified)
    # stop = time.time()
    # print("Total time: ", stop - start)
    return score_with_alpha['avg'], score_with_alpha_modified['avg'], shuffle_split_idxs


def monte_carlo_2v2(X, Y):
    # start = time.time()
    print("Monte-Carlo CV DT Normal")
    # Split into training and testing data
    parameters_ridge = {'alpha': [10000000, 100000000, 1000000000]} #0.01]}#, 0.1, 10, 20, 40, 80, 100, 1000, 10000, 100000, 1000000,
    parameters_dt = {'min_samples_split': [4, 6, 8, 10, 20]}  #



    dt = DecisionTreeRegressor()
    clf = GridSearchCV(dt, param_grid=parameters_dt, scoring='neg_mean_squared_error',
                       refit=True, cv=5, n_jobs=1)

    eeg_features = X# readys_data.iloc[:, :].values  # :208 for thirteen month olds. 208: for nine month olds.
    w2v_embeds_mod = Y  # w2v_embeds[:]  # :208 for thirteen month olds. 208: for nine month olds.

    # print(eeg_features.shape)
    # print(w2v_embeds_mod.shape)
    rs = ShuffleSplit(n_splits=100, train_size=0.90)
    all_data_indices = [i for i in range(len(w2v_embeds_mod))]
    f = 1
    score_with_alpha = {}
    cosine_scores = []
    shuffle_split_idxs = []
    for train_index, test_index in rs.split(all_data_indices):
        # print("Shuffle Split fold: ", f)
        shuffle_split_idxs.append([train_index, test_index])
        X_train, X_test = eeg_features.iloc[train_index,:].values, eeg_features.iloc[test_index,:].values
        # The following two lines are for the permutation test. Comment them out when not using the permutation test.
        # print("Train index before", train_index)
        # random.shuffle(train_index) # For permutation test only.
        # random.shuffle(test_index) # For permutation test only.
        # print("Train index after: ", train_index)
        y_train, y_test = w2v_embeds_mod[train_index], w2v_embeds_mod[test_index]
        # print(X_train.shape)
        # print(y_train.shape)
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

def avg_trials_and_ps_model():
    # First read the data
    print("Avg trials and ps model")
    avg_eeg_data = pickle.load(open(avg_trials_and_ps_9and13_path, 'rb'))
    avg_w2v_data_loaded = load(gen_w2v_all_ps_avg_path)
    avg_w2v_data = avg_w2v_data_loaded['arr_0']
    # print("Avg all ps trial shape: ", avg_eeg_data.shape)
    # print("Avg w2v gen shape: ", avg_w2v_data.shape)
    rounds = 1
    a_scores = []
    p_scores = []
    data_idxs = [t for t in range(len(avg_w2v_data))]
    for r in range(rounds):
        # print("Round: ", r + 1)
        # X_train, X_test, y_train, y_test = train_test_split(avg_eeg_data, avg_w2v_data, train_size=0.90, shuffle=True)
        # # print(X_train.shape)
        # dt_model = Ridge()
        # dt_model.fit(X_train, y_train)
        # preds = dt_model.predict(X_test)
        # a_points, a_total_points, a_score = two_vs_two(y_test, preds)
        # a_scores.append(a_score)
        a_score, shfl_indices = monte_carlo_2v2(avg_eeg_data, deepcopy(avg_w2v_data))
        a_scores.append(a_score)
        random.shuffle(data_idxs)
        mod_avg_w2v_data = avg_w2v_data[data_idxs]
        p_score = monte_carlo_2v2_permuted(avg_eeg_data, deepcopy(mod_avg_w2v_data), shfl_indices)
        p_scores.append(p_score)
    print("Average score for all averaged trials and ps no cv: ", np.average(a_scores))
    print("Average score for all averaged trials and ps no cv permuted: ", np.average(p_scores))

    # avg_w2v_data_idxs = [t for t in range(16)]
    # for r in range(rounds):
    #     random.shuffle(avg_w2v_data_idxs)
    #     avg_w2v_data = avg_w2v_data[avg_w2v_data_idxs]
    #     print("Round: ", r + 1)
    #     X_train, X_test, y_train, y_test = train_test_split(avg_eeg_data, avg_w2v_data, train_size=0.90, shuffle=True)
    #     # print(X_train.shape)
    #     dt_model = Ridge()
    #     dt_model.fit(X_train, y_train)
    #     preds = dt_model.predict(X_test)
    #     a_points, a_total_points, p_score = two_vs_two(y_test, preds)
    #     p_scores.append(p_score)

avg_trials_and_ps_model()

def split_avg_trials_and_ps_model():
    # First read the data
    print("Split avg trial and ps model")
    start = time.time()
    t_a_eeg = pickle.load(open(avg_trials_and_ps_13m_path, 'rb'))
    n_a_eeg = pickle.load(open(avg_trials_and_ps_9m_path, 'rb'))
    avg_w2v_data_loaded = load(gen_w2v_all_ps_avg_path)
    avg_w2v_data = avg_w2v_data_loaded['arr_0']
    rounds = 1
    t_a_scores = []
    n_a_scores = []
    t_a_scores_perm = []
    n_a_scores_perm = []
    data_idxs = [t for t in range(len(avg_w2v_data))]
    for r in range(rounds):
        # t_X_train, t_X_test, t_y_train, t_y_test = train_test_split(t_a_eeg, avg_w2v_data, train_size=0.90, shuffle=True)
        # n_X_train, n_X_test, n_y_train, n_y_test = train_test_split(n_a_eeg, avg_w2v_data, train_size=0.90, shuffle=True)
        #
        # t_a_model = DecisionTreeRegressor()
        # t_a_model.fit(t_X_train, t_y_train)
        # t_a_preds = t_a_model.predict(t_X_test)
        # t_a_points, t_a_total_points, t_a_score = two_vs_two(t_y_test, t_a_preds)
        t_a_score, shfl_indices = monte_carlo_2v2(t_a_eeg, deepcopy(avg_w2v_data))
        t_a_scores.append(t_a_score)
        random.shuffle(data_idxs)
        mod_avg_w2v_data = avg_w2v_data[data_idxs]
        t_a_score_perm = monte_carlo_2v2_permuted(t_a_eeg, deepcopy(mod_avg_w2v_data), shfl_indices)
        t_a_scores_perm.append(t_a_score_perm)

        # n_a_model = DecisionTreeRegressor()
        # n_a_model.fit(n_X_train, n_y_train)
        # n_a_preds = n_a_model.predict(n_X_test)
        # n_a_points, n_a_total_points, n_a_score = two_vs_two(n_y_test, n_a_preds)
        n_a_score, shfl_indices = monte_carlo_2v2(n_a_eeg, deepcopy(avg_w2v_data))
        n_a_scores.append(n_a_score)
        random.shuffle(data_idxs)
        mod_avg_w2v_data = avg_w2v_data[data_idxs]
        n_a_score_perm = monte_carlo_2v2_permuted(n_a_eeg, deepcopy(mod_avg_w2v_data), shfl_indices)
        n_a_scores_perm.append(n_a_score_perm)

    stop = time.time()
    print("Average score for 13 month averaged trials and ps: ", np.average(t_a_scores))
    print("Average score for 9 month averaged trials and ps: ", np.average(n_a_scores))
    print("Average score for 13 month averaged trials and ps permuted: ", np.average(t_a_scores_perm))
    print("Average score for 9 month averaged trials and ps permuted: ", np.average(n_a_scores_perm))
    print("Total time taken: ", stop - start)

# split_avg_trials_and_ps_model()

# def test_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, shuffle=True)
#     model = Ridge()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     a,b,c = two_vs_two(y_test.tolist(), preds)
#     print(c)

# test_file = load("G:\jw_lab\jwlab_eeg\\regression\w2v_embeds\\all_w2v_embeds.npz")
# test_data = test_file['arr_0']
# test_scores_non = []
# test_scores_perm = []
# mds = MDS(n_components=100)
# X = mds.fit_transform(readys_data.iloc[:,:].values)
# for _ in range(10):
#     score = test_model(X, deepcopy(test_data))
#     test_scores_non.append(score)
#     score = test_model_permute(X, deepcopy(test_data))
#     test_scores_perm.append(score)
# print("Non-permuted : ", np.average(test_scores_non))
# print("Permuted : ", np.average(test_scores_perm))


def random_groups():
    start = time.time()
    print("Random Groups all months 20 group factor with SVD")
    all_scores_wo_perm = []
    all_scores_wi_perm = []
    for dummy in range(10):
        print("Iteration: ", dummy)
        grouped_data, grouped_labels = divide_by_labels(readys_data)
        all_grouped_data, all_grouped_labels = random_subgroup(grouped_data, grouped_labels, 20)
        # Now average the groups of data and then combine them.
        data_res, labels_res, meaned_labels = average_grouped_data(all_grouped_data, all_grouped_labels)
        data_res = np.array(data_res)

        embeds_loaded = load(embeds_with_label_path, allow_pickle=True)
        embeds_local = embeds_loaded['arr_0']
        embeds = embeds_local[0]
        y = []
        for label in labels_res:
            y.append(embeds[label])

        y = np.array(y)
        y = do_svd(deepcopy(y))
        wo_perm_score, shuffle_idxs = monte_carlo_2v2(data_res, y)
        # wo_perm_score, wi_perm_score, shuffle_idxs = monte_carlo_2v2_modified(data_res, np.array(y))
        # Here, each group of the labels for the averaged samples is the given the same value.

        key = list(set(labels_res))
        random.shuffle(key)
        new_meaned_labels = deepcopy(meaned_labels)
        for o in range(len(meaned_labels)):
            for p in range(len(meaned_labels[o])):
                new_meaned_labels[o][p] = key[o]


        permuted_new_labels = []
        for l in new_meaned_labels:
            for m in l:
                permuted_new_labels.append(m.tolist())

        y = []
        for label in permuted_new_labels:
            y.append(embeds[label])

        y = np.array(y)
        y = do_svd(deepcopy(y))

        wi_perm_score = monte_carlo_2v2_permuted(data_res, y, shuffle_idxs)
        all_scores_wo_perm.append(wo_perm_score)
        all_scores_wi_perm.append(wi_perm_score)

    print("Averaged scores after rerandomization non-permuted", np.average(all_scores_wo_perm))
    print("Averaged scores after rerandomization permuted", np.average(all_scores_wi_perm))
    stop = time.time()
    print("Total time taken: ", stop - start)


# random_groups()


monte_carlo_2v2()

# split_ps_model()



