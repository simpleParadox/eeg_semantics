import pandas as pd
import numpy as np
from jwlab.data_graph import plot_good_trial_participant, plot_good_trial_word
from jwlab.participants_map import map_participants
from jwlab.constants import word_list
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word
from scipy.signal import resample
from jwlab.ml_prep import  average_trials_and_participants, no_average, average_trials



sliding_window_time_length = [300, 400, 500, 600]




def load_ml_data(filepath, participants):
    # read all participant csvs, concat them into one dataframe
    dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (filepath, s))
           for s in participants]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    ys = [np.loadtxt("%s%s_labels.txt" % (filepath, s)).tolist()
          for s in participants]
    #print("loaded", flush=True)
    return df, ys


def prep_ml_graphs_first20(filepath, participants, downsample_num=1000, averaging="average_trials"):
    df, ys = load_ml_data(filepath, participants)
    return prep_ml_internal(df, ys, participants, downsample_num=downsample_num, averaging=averaging)


def sliding_window(df, time_length):
    df_list = []
    for i in range(0, 1100-time_length, 100):
        df_list.append(df[(df.Time < time_length + i) & (df.Time >= i)])
    return df_list

def prep_ml_internal_no_sliding_window(df, ys, participants, downsample_num=1000, averaging="average_trials"):
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

    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65"], axis=1)

    # now we need to flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    X = np.reshape(X, (1000, 60, -1))
    X = resample(X, downsample_num, axis=0)
    (i, j, k) = X.shape
    X = np.reshape(X, (k, j * downsample_num))

    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    df = pd.DataFrame(data=X)
    df['label'] = Y
    df['participant'] = np.concatenate([[ys.index(y)]*len(y) for y in ys])

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

def prep_ml_internal_depreciated(df, ys, participants, downsample_num=1000, averaging="average_trials"):
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

    # print good trial each participant graph
    good_trial_participant_count = np.around(np.true_divide(
        np.subtract(trial_count, bad_trial_count), trial_count), decimals=2)
    good_trial_word_count = get_left_trial_each_word(participants)

    Y = np.concatenate(ys)

    dfs = []
    dfs.append(df[(df.Time<100) & (df.Time>=0)])
    dfs.append(df[(df.Time<200) & (df.Time>=0)])
    dfs.append(df[(df.Time<200) & (df.Time>=100)])

    X_list = [0 for i in range(len(dfs))]
    y_list = [0 for i in range(len(dfs))]
    p_list = [0 for i in range(len(dfs))]
    w_list = [0 for i in range(len(dfs))]
    df_list = [0 for i in range(len(dfs))]

    for each_window in range(len(dfs)):
        df = dfs[each_window]
        # we don't want the time column, or the reference electrode, so drop those columns
        df = df.drop(columns=["Time", "E65"], axis=1)

        # now we need to flatten each
        # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
        # of 64*1000 columns of eeg data
        X = df.values
        if each_window == 0 or each_window == 2:
            X = np.reshape(X, (100, 60, -1))
        else:
            X = np.reshape(X, (200, 60, -1))
            
        X = resample(X, downsample_num, axis=0)
        (i, j, k) = X.shape
        X = np.reshape(X, (k, j * downsample_num))

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

        X_list[each_window] = X
        y_list[each_window] = y
        p_list[each_window] = p
        w_list[each_window] = w
        df_list[each_window] = df

    return X_list, y_list, [good_trial_participant_count, good_trial_word_count]

def prep_ml_internal(df, ys, participants, downsample_num=1000, averaging="average_trials"):
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

    # sliding window - sw_list_for_all_time_length[len(time_length)][1000-time_length/100]
    sw_list_for_all_time_length = []
    for time_length in sliding_window_time_length:
        sw_list_for_all_time_length.append(sliding_window(df, time_length))

    X_list = [[0 for i in range(int((1100-sliding_window_time_length[j])/100))]
              for j in range(len(sliding_window_time_length))]
    y_list = [[0 for i in range(int((1100-sliding_window_time_length[j])/100))]
              for j in range(len(sliding_window_time_length))]
    p_list = [[0 for i in range(int((1100-sliding_window_time_length[j])/100))]
              for j in range(len(sliding_window_time_length))]
    w_list = [[0 for i in range(int((1100-sliding_window_time_length[j])/100))]
              for j in range(len(sliding_window_time_length))]
    df_list = [[0 for i in range(int((1100-sliding_window_time_length[j])/100))]
               for j in range(len(sliding_window_time_length))]

    for each_timeLength in range(len(sw_list_for_all_time_length)):
        for each_df in range(len(sw_list_for_all_time_length[each_timeLength])):
            df = sw_list_for_all_time_length[each_timeLength][each_df]
            # we don't want the time column, or the reference electrode, so drop those columns
            df = df.drop(columns=["Time", "E65"], axis=1)

            # now we need to flatten each
            # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
            # of 64*1000 columns of eeg data
            X = df.values
            X = np.reshape(
                X, (sliding_window_time_length[each_timeLength], 60, -1))
            #X = resample(X, downsample_num, axis=0)
            (i, j, k) = X.shape
            X = np.reshape(X, (k, j * sliding_window_time_length[each_timeLength]))

            # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
            df = pd.DataFrame(data=X)
            df['label'] = Y
            df['participant'] = np.concatenate(
                [[ys.index(y)]*len(y) for y in ys])
            
            # remove bad samples
            df = df[df.label != -1]

            # make label zero indexed
            df.label -= 1
            
            # get the first 20 rows of each participant
            df = df.groupby('participant').head(20)
                
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

            X_list[each_timeLength][each_df] = X
            y_list[each_timeLength][each_df] = y
            p_list[each_timeLength][each_df] = p
            w_list[each_timeLength][each_df] = w
            df_list[each_timeLength][each_df] = df

    return X_list, y_list, [good_trial_participant_count, good_trial_word_count]


# Raw data




def create_ml_df(filepath, participants, downsample_num=1000):
    df, ys = load_ml_data(filepath, participants)
    return create_ml_df_internal_sktime(df, ys, participants, downsample_num=downsample_num)


def create_ml_df_internal(df, ys, participants, downsample_num=1000):
    # for the ml segment we only want post-onset data, ie. sections of each epoch where t>=0
    df = df[df.Time >= 0]
    # we don't want the time column, or the reference electrode, so drop those columns
    df = df.drop(columns=["Time", "E65", "E64", "E63", "E62", "E61"], axis=1)

    # now we need to flatten each
    # "block" of data (ie. 1000 rows of 64 columns of eeg data) into one training example, one row
    # of 64*1000 columns of eeg data
    X = df.values
    X = np.reshape(X, (1000, 60, -1))
    X = resample(X, downsample_num, axis=0)
    (i, j, k) = X.shape
    X = np.reshape(X, (k, j * downsample_num))

    # map first participants (cel from 1-4ß map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants)
    ys = map_participants(ys, participants)
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1
    y = np.concatenate(ys)

    assert y.shape[0] == X.shape[0]
    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    df = pd.DataFrame(data=X)
    df['label'] = y
    df['participant'] = np.concatenate([[i]*len(y) for i, y in enumerate(ys)])

    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed
    df.label -= 1

    return df

# --- BEING USED ON COMPUTE CANADA ---


def create_ml_df_internal_sktime(df, ys, participants, downsample_num=1000):
    df = df[df.Time >= 0]
    df = df.drop(columns=["Time", "E65", "E64", "E63", "E62", "E61"], axis=1)

    df['id'] = np.concatenate(
        [[i] * 1000 for i in range(len(df.index) // 1000)])
    df = df.groupby(["id"], as_index=False).agg(pd.Series)

    # map first participants (cel from 1-4 map to 1-16), then concatenate all ys, and ensure the sizes are correct
    ybad = get_bad_trials(participants)
    ys = map_participants(ys, participants)
    for each_ps in range(len(ys)):
        for bad_trial in range(len(ybad[each_ps])):
            ys[each_ps][ybad[each_ps][bad_trial]-1] = -1
    y = np.concatenate(ys)

    # make new dataframe where each row is now a sample, and add the label and particpant column for averaging
    # df = pd.DataFrame(data=X)
    df['label'] = y
    df['participant'] = np.concatenate([[ys.index(y)]*len(y) for y in ys])

    # remove bad samples
    df = df[df.label != -1]

    # make label zero indexed
    df.label -= 1
    return df


def save_ml_df(df, filepath):
    df.to_pickle(filepath)


def load_ml_df(filepath):
    return pd.read_pickle(filepath)


def y_to_binary(y):
    y[y < 8] = 0
    y[y >= 8] = 1
    return y