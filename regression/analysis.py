import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import gensim
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
# from sklearn.svm import LinearSVR
# from sklearn.svm import SVR
# from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


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
    from regression import rsa_helper
else:
    from functions import average_trials, average_trials_and_participants, labels_mapping, two_vs_two, test_model, test_model_permute, two_vs_two_test, divide_by_labels, random_subgroup, average_grouped_data, get_w2v_embeds
    import rsa_helper

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
f = open(readys_path, 'rb')
readys_data = pickle.load(f)
f.close()




ps = 13  # [12 months -> 0-12 (indices 0-1007), 9 months -> 13-end (indices 1008-end)]
word = 2
group = 12

ps2 = 14

months_12 = [0,1,2,3,4,5,6,7,8,9,11,12] # PS 10 has almost nothing.
months_9 = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]

# max_rdm_corr_groups = 1.6268522741479599

# df, ps, word = rsa_helper.eeg_filter_subject(deepcopy(readys_data), ps, word)  # Returns data for a subject for a specific word.
# df, parts, ps, word = rsa_helper.eeg_filter_by_group(readys_data, group,word)  # Filter out data for different months - for single word. For example. 9 month olds - word 0, 12 month olds word 1, etc.
df, ps, word = rsa_helper.eeg_filter_subject_all_words(readys_data, 10)  # Filter out all the words averaged together for a single participants
# df, group, ps, word = rsa_helper.eeg_filter_by_group_all_words(readys_data, group)
#


labels = [0,1,4,6,7,15]
## Averaging RDMs
nan_set = set()
for i in months_9:
    df, ps, word = rsa_helper.eeg_filter_subject_all_words(readys_data, i)#, deepcopy(labels))
    # print(df.shape)

    for j in range(df.shape[0]):
        if not (sum(df[j]) > 0 or sum(df[j]) < 0):
            print(ps," ", j)
            nan_set.add(j)

rdm_list = []
for i in months_9:
    df, ps, word = rsa_helper.eeg_filter_subject_all_words(readys_data, i)#, deepcopy(labels))
    df = np.delete(df, list(nan_set), axis=0)
    print(df.shape)
    rdm = rsa_helper.eeg_rdm_dist_corr(df)
    rdm_list.append(rdm)

rdm_list = np.array(rdm_list)
t = np.mean(rdm_list, axis=0)
rsa_helper.RDM_vis(t, 'all 9 months', 'present in both')

# Words not present for 12 month olds





#     for j in range(df.shape[0]):
#         if not sum(df[j]) > 0:
#             nan_set.add(j)
#             # df = np.delete(df, j, 0)
#
# for i in range(0,16):
#     df, ps, word = rsa_helper.eeg_filter_subject_all_words(readys_data, ps)
#     df = np.delete(df, list(nan_set), axis=0)
#     print(df.shape)

# # print(parts)
# df = df[~np.isnan(df).any(axis=1)]  # removes nan values from the numpy array.
# rdm = rsa_helper.eeg_rdm_dist_corr(df)
#
#
# rsa_helper.RDM_vis(rdm, ps, word)



# ## For two subjects df -> to test, whether the 9 months are just sitting around doing nothing.
# df1, ps1, word = rsa_helper.eeg_filter_subject_all_words(readys_data, ps)
# df2, ps2, word = rsa_helper.eeg_filter_subject_all_words(readys_data, ps2)
# rdm = rsa_helper.alternate_ps_eeg_corr(df1, df2)
# rsa_helper.RDM_vis(rdm, ps, word, ps1, ps2)


# Plotting some graphs

# Let's plot a histogram about the distribution.

# plt.hist(df)
# plt.show()

