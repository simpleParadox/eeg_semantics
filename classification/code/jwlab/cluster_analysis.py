import pandas as pd
import numpy as np
from numpy import copy
from jwlab.constants import cleaned_data_filepath
from jwlab.ml_prep import  average_trials_and_participants, no_average, average_trials, no_average_labels
from jwlab.data_graph import plot_good_trial_participant, plot_good_trial_word
from jwlab.participants_map import map_participants
from jwlab.constants import word_list, bad_trials_filepath, old_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, RepeatedKFold
from scipy import stats
import more_itertools as mit
from IPython.display import display
import math
from copy import deepcopy

################################ Restructure Matrix ################################
def permutation_and_average(df, avg_trial):
    #display(df)
    #print(df.groupby(['label']).size().reset_index(name='Count'))
    grouped = df.groupby(['label'])
    shuffled_dfs = []
    for i in range(16):
        # df_ith has all trials with word i 
        df_ith = grouped.get_group(i)
        # randomize the rows
        df_ith = df_ith.sample(frac=1).reset_index(drop=True)
        display(df_ith)
        
        orig_df_len = len(df_ith)
        expected_df_len = 0
        
        # The leftover trials
        leftover = len(df_ith) % avg_trial
        halfsize_set = math.ceil(avg_trial/2)
        
        # If the number of leftover trials is less than half size of a group, then we put them into the last group  
        if leftover < halfsize_set:
            # Expected df_len after averaging should be floor(orig_df_len/avg_trial)
            expected_df_len = orig_df_len // avg_trial
            
            last_group_df = df_ith.tail(leftover+avg_trial)
            last_group_mean = last_group_df.mean(axis=0)
            
            # All other rows except for ones in the last group 
            df_ith_temp = df_ith.head(-(leftover+avg_trial))
            df_ith = df_ith_temp.set_index(np.arange(len(df_ith_temp)) // avg_trial).mean(level=0)
            
            # Append everything else with the last group
            df_ith = df_ith.append(last_group_mean, ignore_index=True)
        else: # Else if the number of leftover trials is more than half of a group, they form a group themselves
            # Expected df_len after averaging should be ceil(orig_df_len/avg_trial)
            expected_df_len = math.ceil(orig_df_len/avg_trial)
            
            df_ith = df_ith.set_index(np.arange(len(df_ith)) // avg_trial).mean(level=0)
            
        display(df_ith)
        assert len(df_ith) == expected_df_len
        shuffled_dfs += [df_ith]
        
    # Append shuffled dataframes of each word together
    result = pd.concat(shuffled_dfs)
    #display(result)
    return result

################################ prep data ################################

def slide_df(df, length_per_window):
    num_windows = int(1200 / length_per_window) #changed from 1000
    windows_list = []
    first_range = int(200 / length_per_window) 
    for i in range(first_range, 0, -1):
          windows_list.append(df[(df.Time < -(i-1) * length_per_window) & (df.Time >= -i * length_per_window)])
    second_range= int(1000 /length_per_window)
    for i in range(second_range):
        windows_list.append(df[(df.Time < ((i+1) * length_per_window)) & (df.Time >= i * length_per_window)])
    assert len(windows_list) == num_windows
    return windows_list

def load_ml_data(filepath, participants):
    # print("First line load ml data")
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s))
           for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
    # print("Load ml data")
    # print(df)
    # print(df.shape)
    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)).tolist()
          for s in participants]
    #print("loaded", flush=True)
    return df, ys

def prep_cluster_analysis_permutation(filepath, participants, downsample_num=1000, length_per_window=10, useRandomizedLabel=False):
    df, ys = load_ml_data(filepath, participants)
    return prep_cluster_analysis_internal_permutation(df, ys, participants, downsample_num=downsample_num, length_per_window=length_per_window, useRandomizedLabel=useRandomizedLabel)


def prep_cluster_analysis(filepath, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=10, useRandomizedLabel=False):
    df, ys = load_ml_data(filepath, participants)
    print("Randomized Labels?", useRandomizedLabel)
    return prep_cluster_analysis_internal(df, ys, participants, downsample_num=downsample_num, averaging=averaging, length_per_window=length_per_window, useRandomizedLabel=useRandomizedLabel)


def prep_cluster_analysis_internal_permutation(df, ys, participants, downsample_num=1000, length_per_window=10, useRandomizedLabel=False):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]

    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants)
    ys = map_participants(ys, participants)

    # set the value of bad trials in ys_curr to -1 (to exclude from learning)
    trial_count = []
    bad_trial_count = []
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            # ys_curr[each_ps]: for the total trial sub-list of each participant of ys_curr...
            # ybad[each_ps][bad_trial]: for each trial index in the bad trial sub-list of each participant of ybad...
            # minus 1 since in ys_curr trials are zero-indexed while in bad_trial it's one-indexed (because they are directly read from csv)
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1

        # count the total number of trials for each participant
        trial_count += [len(ys[each_ps])]
        bad_trial_count += [len(ybad[each_ps])]

    # good trial each participant 
    good_trial_participant_count = np.around(np.true_divide(
        np.subtract(trial_count, bad_trial_count), trial_count), decimals=2)
    # good trial each word each participant
    good_trial_word_count = get_left_trial_each_word(participants)

    Y = np.concatenate(ys)
    
    if useRandomizedLabel:
        random.shuffle(Y)
        print("labels shuffled")

    windows_list = slide_df(df, length_per_window)

    X_list = [0] * int(1000 / length_per_window)
    y_list = X_list[:]
    p_list = X_list[:]
    w_list = X_list[:]
    df_list = X_list[:]
    
    for each_window in range(len(windows_list)):
        df = windows_list[each_window]
        df = df.drop(columns=["Time", "E65"], axis=1)
        X = df.values
        X = np.reshape(
            X, (length_per_window, 60, -1))
        #X = resample(X, downsample_num, axis=0)
        (i, j, k) = X.shape
        X = np.reshape(X, (k, j * length_per_window))

        # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
        df = pd.DataFrame(data=X)
        df['label'] = Y
        df['participant'] = np.concatenate(
            [[ys.index(y)]*len(y) for y in ys])

        # remove bad samples
        df = df[df.label != -1]

        # make label zero indexed
        df.label -= 1
        
        # At this step, we change the matrix structure
        df = permutation_and_average(df, 5)

        X, y, p, w = no_average(df)

        y[y < 8] = 0
        y[y >= 8] = 1
        
        X_list[each_window] = X
        y_list[each_window] = y
        p_list[each_window] = p
        w_list[each_window] = w
        df_list[each_window] = df
        
    return X_list, y_list, [good_trial_participant_count, good_trial_word_count]

def remap_label(y):
    print("Remap label")
    labels_temp = list(range(0, 16))
    random.shuffle(labels_temp)

    mapdict = {}
    for i in range(16):
        mapdict[i] = labels_temp.index(i)

    newArray = copy(y)

    for k, v in mapdict.items():
        newArray[y == k] = v

    # np.random.shuffle(newArray)
    return newArray

def prep_cluster_analysis_internal(df, ys, participants, downsample_num=1200, averaging="average_trials_and_participants", length_per_window=10, useRandomizedLabel=False):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    # df = df[df.Time >= 0] #removed

    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants)
    ys = map_participants(ys, participants) # makes a 1 -16 array for each trial type

    # set the value of bad trials in ys_curr to -1 (to exclude from learning)
    trial_count = []
    bad_trial_count = []
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            # ys_curr[each_ps]: for the total trial sub-list of each participant of ys_curr...
            # ybad[each_ps][bad_trial]: for each trial index in the bad trial sub-list of each participant of ybad...
            # minus 1 since in ys_curr trials are zero-indexed while in bad_trial it's one-indexed (because they are directly read from csv)
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1  # so now 0 to 15

        # count the total number of trials for each participant
        trial_count += [len(ys[each_ps])]
        bad_trial_count += [len(ybad[each_ps])]

    # good trial each participant 
    good_trial_participant_count = np.around(np.true_divide(
        np.subtract(trial_count, bad_trial_count), trial_count), decimals=2)
    # good trial each word each participant
    good_trial_word_count = get_left_trial_each_word(participants)

    Y = np.concatenate(ys)

    if useRandomizedLabel:
        random.shuffle(Y)
        np.random.shuffle(Y)
        # Y = remap_label(Y)
        # print("Labels shuffled.")

    windows_list = slide_df(df, length_per_window)

    X_list = [0] * int(1200 / length_per_window) # changed to post length
    y_list = X_list[:]
    p_list = X_list[:]
    w_list = X_list[:]
    df_list = X_list[:]
    
    for each_window in range(len(windows_list)):
        df = windows_list[each_window]
        df = df.drop(columns=["Time", "E65"], axis=1)
        X = df.values
        X = np.reshape(
            X, (length_per_window, 60, -1))
        #X = resample(X, downsample_num, axis=0)
        (i, j, k) = X.shape
        X = np.reshape(X, (k, j * length_per_window))
        # print(X.shape)

      
        

        # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
        df = pd.DataFrame(data=X)
        df['label'] = Y
        df['participant'] = np.concatenate(
            [[ys.index(y)]*len(y) for y in ys])

        # remove bad samples
        df = df[df.label != -1]

        # make label zero indexed
        df.label -= 1
        
#         # get the first 20 rows of each participant
#         df = df.groupby('participant').head(20)

        # different averaging processes
        if averaging == "no_averaging":
            X, y, p, w = no_average(df)
        elif averaging == "average_trials":
            X, y, p, w = average_trials(df)
        elif averaging == "average_trials_and_participants":
            X, y, p, w = average_trials_and_participants(df, participants)
        elif averaging == "no_average_labels":
            X, y, p, w = no_average_labels(df)
        else:
            raise ValueError("Unsupported averaging!")

        y[y < 8] = 0
        y[y >= 8] = 1
        
        X_list[each_window] = X
        y_list[each_window] = y
        p_list[each_window] = p
        w_list[each_window] = w
        df_list[each_window] = df
                
    return X_list, y_list, [good_trial_participant_count, good_trial_word_count]


def prep_raw_pred_avg(X, participants, length_per_window, num_sliding_windows, test_size=0.2):
    # set up for new df with labels and ps
    num_participants = len(participants)
    num_indices = len(X[0])
    fivefold_testsize = int(test_size*num_indices)
    test_indices = np.random.choice(num_indices-1, fivefold_testsize, replace=False)
    
#     threefold_testsize = int(.33*num_indices)
#     test_indices = np.random.choice(num_indices-1, treefold_testsize, replace=False)

    df_test = []
    df_train = []

    for i in range(num_sliding_windows):
        ## will need each window
        X[i] = X[i].reset_index()   
        
        # #create new df with these indices and removing from orig
        df_test.append(X[i].iloc[test_indices])
        df_train.append(X[i].drop(X[i].index[test_indices]))
        assert(len(df_train[i]) + len(df_test[i]) == len(X[i]))
        df_test[i] = df_test[i].drop(columns=['index'], axis=1) 
        df_train[i] = df_train[i].drop(columns=['index'], axis=1)



    # create training matrix:
    X_train=[]
    y_train=[]
    for i in range(num_sliding_windows):
        y_train.append(df_train[i].label.values)
        X_train.append(df_train[i].drop(columns = ['label', 'participant'], axis = 1))

    # ## Creating a copy of the training labels. Added by Rohan.
    # y_train_labels = deepcopy(y_train)

    # create test matrices
    X_test = [] # test raw trials
    y_test = [] 
    X_test_t = [] # test avg trials
    y_test_t = [] 
    X_test_pt = [] # test avg trials and ps
    y_test_pt = [] 

    for i in range(num_sliding_windows):
        y_test.append(df_test[i].label.values)
        X_test.append(df_test[i].drop(columns = ['label', 'participant'], axis = 1))

        X_test_t_temp, y_test_temp, ps, w = average_trials(df_test[i])
        X_test_t.append(pd.DataFrame(X_test_t_temp))
        y_test_t.append(y_test_temp)

        X_test_pt_temp, y_test_temp_pt, ps, w = average_trials_and_participants(df_test[i], participants)
        X_test_pt.append(pd.DataFrame(X_test_pt_temp))
        y_test_pt.append(y_test_temp_pt)

        

        ## If you want the raw labels itself, just comment the following 8 lines of code. Commented by Rohan for regression purposes.
        # y_train[i][y_train[i] < 8] = 0
        # y_train[i][y_train[i] >= 8] = 1
        #
        # y_test[i][y_test[i] < 8] = 0
        # y_test[i][y_test[i] >= 8] = 1
        #
        # y_test_t[i][y_test_t[i] < 8] = 0
        # y_test_t[i][y_test_t[i] >= 8] = 1
        #
        # y_test_pt[i][y_test_pt[i] < 8] = 0
        # y_test_pt[i][y_test_pt[i] >= 8] = 1



        
        # to predict the letter b (word onset)
#     for i in range(num_sliding_windows):
#         y_train[i][y_train[i] < 4] = 0 # b
#         y_train[i][y_train[i] == 8] = 0 # b
#         y_train[i][y_train[i] == 9] = 0 # b
#         y_train[i][y_train[i] == 4] = 1
#         y_train[i][y_train[i] == 5] = 1
#         y_train[i][y_train[i] == 6] = 1
#         y_train[i][y_train[i] == 7] = 1
#         y_train[i][y_train[i] > 9] = 1

#         y_test[i][y_test[i] < 4] = 0 # b
#         y_test[i][y_test[i] == 8] = 0 # b
#         y_test[i][y_test[i] == 9] = 0 # b
#         y_test[i][y_test[i] == 4] = 1
#         y_test[i][y_test[i] == 5] = 1
#         y_test[i][y_test[i] == 6] = 1
#         y_test[i][y_test[i] == 7] = 1
#         y_test[i][y_test[i] > 9] = 1

#         y_test_t[i][y_test_t[i] < 4] = 0 # b
#         y_test_t[i][y_test_t[i] == 8] = 0 # b
#         y_test_t[i][y_test_t[i] == 9] = 0 # b
#         y_test_t[i][y_test_t[i] == 4] = 1
#         y_test_t[i][y_test_t[i] == 5] = 1
#         y_test_t[i][y_test_t[i] == 6] = 1
#         y_test_t[i][y_test_t[i] == 7] = 1
#         y_test_t[i][y_test_t[i] > 9] = 1

#         y_test_pt[i][y_test_pt[i] < 4] = 0 # b
#         y_test_pt[i][y_test_pt[i] == 8] = 0 # b
#         y_test_pt[i][y_test_pt[i] == 9] = 0 # b
#         y_test_pt[i][y_test_pt[i] == 4] = 1
#         y_test_pt[i][y_test_pt[i] == 5] = 1
#         y_test_pt[i][y_test_pt[i] == 6] = 1
#         y_test_pt[i][y_test_pt[i] == 7] = 1
#         y_test_pt[i][y_test_pt[i] > 9] = 1
        
        
    return X_train, y_train, X_test, y_test, X_test_t, y_test_t, X_test_pt, y_test_pt


################################ Analysis procedure ################################

def init(age_group):
    length_per_window = 10
    num_sliding_windows = int(1000/ length_per_window)
    num_folds = 3
    num_iterations = 10
    
    if age_group is 9:
        participants = ["904", "905", "906", "908", "909", "912","913", "914", "916", "917", "919",\
                        "920", "921", "923", "924", "927", "928", "929", "930", "932"]
    elif age_group is 11:
        participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]
    else:
        raise ValueError("Unsupported age group!")
        
    return length_per_window, num_sliding_windows, num_folds, num_iterations, participants

def prep_data(participants, length_per_window, useRandomizedLabel):
    X, y, good_trial_count = prep_cluster_analysis(cleaned_data_filepath, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=length_per_window, useRandomizedLabel=useRandomizedLabel)
    
    return X, y

def cross_validaton(num_iterations, num_sliding_windows, num_folds, X, y):
    results = {}
    rkf = RepeatedKFold(n_splits=3, n_repeats=10)
    
    for i in range(num_sliding_windows):
        if num_sliding_windows > 1:
            X_temp = X[i]
            y_temp = y[i]
        else:
            X_temp = X
            y_temp = y
        
        for train_index, test_index in rkf.split(X_temp):
            X_train, X_test = X_temp[train_index], X_temp[test_index]
            y_train, y_test = y_temp[train_index], y_temp[test_index]
            
            #model = SVC(kernel = 'rbf')
            model = LinearSVC(C=1e-9, max_iter=5000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            testScore = accuracy_score(y_test,y_pred)
            
            if i in results.keys(): 
                results[i] += [testScore]
            else:
                results[i] = [testScore]
                
    for i in range(num_sliding_windows):
        assert len(results[i]) == num_iterations * num_folds
    
    return results

def t_test(results, num_iterations, num_sliding_windows, num_folds):
    pvalues = []
    for i in range(num_sliding_windows):
        istat = stats.ttest_1samp(results[i], .5)
        pvalues += [istat.pvalue] if istat.statistic > 0 else [1]
    
    return pvalues

# Finding contiguous time cluster
def find_clusters(pvalues):
    valid_window = [i for i,v in enumerate(pvalues) if v <= 0.025]
    print("Valid windows are: {0}\n".format(valid_window))
    
    # Obtain clusters (3 or more consecutive meaningful time)
    clusters = [list(group) for group in mit.consecutive_groups(valid_window)]
    clusters = [group for group in clusters if len(group) >= 3]
    print("Clusters are: {0}\n".format(clusters))
    
    return clusters

def get_max_t_mass(clusters, pvalues):
    t_mass = []
    for c in clusters:
        t_scores = 0
        for time in c:
            t_scores += pvalues[time]
        t_mass += [t_scores]
    
    max_t_mass = max(t_mass)
    print("The max t mass is: {0}\n".format(max_t_mass))
    
    return max_t_mass

def cluster_analysis_procedure(age_group, useRandomizedLabel):
    length_per_window, num_sliding_windows, num_folds, num_iterations, participants = init(age_group)
    
    X, y = prep_data(participants, length_per_window, useRandomizedLabel)
    
    results = cross_validaton(num_iterations, num_sliding_windows, num_folds, X, y)
    
    pvalues = t_test(results, num_iterations, num_sliding_windows, num_folds)
    
    clusters = find_clusters(pvalues)
    
    max_t_mass = get_max_t_mass(clusters, pvalues)
    
    return max_t_mass

################################ Permutation ################################
def prep_data_permutation(participants, length_per_window, useRandomizedLabel):
    X, y, good_trial_count = prep_cluster_analysis_permutation(cleaned_data_filepath, participants, downsample_num=1000, length_per_window=length_per_window, useRandomizedLabel=useRandomizedLabel)
    return X, y
    
def cluster_analysis_procedure_permutation(age_group, useRandomizedLabel):
    length_per_window, num_sliding_windows, num_folds, num_iterations, participants = init(age_group)

    X, y = prep_data_permutation(participants, length_per_window, useRandomizedLabel)

    results = cross_validaton(num_iterations, num_sliding_windows, num_folds, X, y)

    pvalues = t_test(results, num_iterations, num_sliding_windows, num_folds)

    clusters = find_clusters(pvalues)

    max_t_mass = get_max_t_mass(clusters, pvalues)

    return max_t_mass

################################ Temp ################################

def prep_cluster_analysis_610_1000(filepath, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=10, useRandomizedLabel=False):
    df, ys = load_ml_data(filepath, participants)
    return prep_cluster_analysis_internal_610_1000(df, ys, participants, downsample_num=downsample_num, averaging=averaging, length_per_window=length_per_window, useRandomizedLabel=useRandomizedLabel)

def prep_cluster_analysis_internal_610_1000(df, ys, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=10, useRandomizedLabel=False):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    
    # we only want 610 - 1000ms
    df = df[df.Time >= 610]
    
    window_length = 1000 - 610

    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants)
    ys = map_participants(ys, participants)

    # set the value of bad trials in ys_curr to -1 (to exclude from learning)
    trial_count = []
    bad_trial_count = []
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            # ys_curr[each_ps]: for the total trial sub-list of each participant of ys_curr...
            # ybad[each_ps][bad_trial]: for each trial index in the bad trial sub-list of each participant of ybad...
            # minus 1 since in ys_curr trials are zero-indexed while in bad_trial it's one-indexed (because they are directly read from csv)
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1

        # count the total number of trials for each participant
        trial_count += [len(ys[each_ps])]
        bad_trial_count += [len(ybad[each_ps])]

    # good trial each participant 
    good_trial_participant_count = np.around(np.true_divide(
        np.subtract(trial_count, bad_trial_count), trial_count), decimals=2)
    # good trial each word each participant
    good_trial_word_count = get_left_trial_each_word(participants)

    Y = np.concatenate(ys)
    
    if useRandomizedLabel:
        random.shuffle(Y)
        print("labels shuffled")
    
    df = df.drop(columns=["Time", "E65"], axis=1)
    X = df.values
    X = np.reshape(
        X, (window_length, 60, -1))
    #X = resample(X, downsample_num, axis=0)
    (i, j, k) = X.shape
    X = np.reshape(X, (k, j * window_length))

    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    df = pd.DataFrame(data=X)
    df['label'] = Y
    df['participant'] = np.concatenate(
        [[ys.index(y)]*len(y) for y in ys])

    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed
    df.label -= 1

    # different averaging processes
    if averaging == "no_averaging":
        X, y, p, w = no_average(df)
    elif averaging == "average_trials":
        X, y, p, w = average_trials(df)
    elif averaging == "average_trials_and_participants":
        X, y, p, w = average_trials_and_participants(df, participants)
    else:
        raise ValueError("Unsupported averaging!")

    y[y < 8] = 0
    y[y >= 8] = 1
                
    return X, y, [good_trial_participant_count, good_trial_word_count]

def init_610_1000(age_group):
    length_per_window = 1000 - 610
    num_sliding_windows = 1
    num_folds = 3
    num_iterations = 10
    
    if age_group is 9:
        participants = ["904", "905", "906", "908", "909", "912","913", "914", "916", "917", "919",\
                    "920", "921", "923", "924", "927", "928", "929", "930", "932"]
    elif age_group is 11:
        participants = ["105", "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]
    else:
        raise ValueError("Unsupported age group!")
        
    return length_per_window, num_sliding_windows, num_folds, num_iterations, participants

####### 610 - 1000ms #######
def prep_data_610_1000(participants, length_per_window, useRandomizedLabel):
    X, y, good_trial_count = prep_cluster_analysis_610_1000(cleaned_data_filepath, participants, downsample_num=1000, averaging="average_trials_and_participants", length_per_window=length_per_window, useRandomizedLabel=useRandomizedLabel)
    print(X.shape)
    return X, y

# 610 - 1000ms
def cluster_analysis_procedure_610_1000(age_group, useRandomizedLabel):
    length_per_window, num_sliding_windows, num_folds, num_iterations, participants = init_610_1000(age_group)

    X, y = prep_data_610_1000(participants, length_per_window, useRandomizedLabel)

    results = cross_validaton(num_iterations, num_sliding_windows, num_folds, X, y)

    pvalues = t_test(results, num_iterations, num_sliding_windows, num_folds)

    clusters = find_clusters(pvalues)

    max_t_mass = get_max_t_mass(clusters, pvalues)