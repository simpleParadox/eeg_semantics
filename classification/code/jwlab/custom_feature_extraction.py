import pandas as pd
import numpy as np
from scipy.stats import entropy, kurtosis, skew
from functools import reduce


def extract_features(df):
    df = compress_rows(df)
    
    funcs = [np.mean, np.min, np.max, np.var, skew, kurtosis]
    time_windows = [50, 150, 200, 250, 300, 350, 400, 450, 550, 650, 750, 1000]  
    
    trial_id = df.trial_id
    label = df.label
    participant = df.participant
    df = df.drop(columns=["trial_id", "label", "participant"], axis=1)

    def window(X, index):    
        return X[time_windows[index]:time_windows[index + 1]]

    df_windowed = [df.applymap(lambda x: window(x, i)) for i in range(len(time_windows) - 1)]
    df_extracted = [[cdf.applymap(f) for f in funcs] for cdf in df_windowed]
    df_extracted_flat = [item for sublist in df_extracted for item in sublist]
    for i, df_to_suffix in enumerate(df_extracted_flat):
        df_extracted_flat[i] = df_to_suffix.add_suffix("_%d" % i)
    
    def join_dfs(a, b):
        return a.join(b)

    df_extracted_all_concat = reduce(join_dfs, df_extracted_flat)
    df_extracted_all_concat['trial_id'] = trial_id
    df_extracted_all_concat['label'] = label
    df_extracted_all_concat['participant'] = participant
    return df_extracted_all_concat

def compress_rows(df):
    # compress all 1000 rows into a single row with the same number of columns
    # where each element is now a series of all 1000 data points
    
    # the width of the new dataframe will be 60 electrode data columns
    # + 1 for the participant + 1 for the trial id + 1 for label,
    # and the height is the number of unique trial ids
    trial_ids = np.unique(df.trial_id)
    w, h = 60 + 3, len(trial_ids)
    new_data = [[None for x in range(w)] for y in range(h)]

    for y, t_id in enumerate(trial_ids):
        # extract only the rows for this trial id
        t_dp = df[df.trial_id == t_id]
        for x in range(60):
            # then for each column, pull out the E1/E2/.../E60 electrode data, put it all in a series,
            # and then plop that in a single cell
            new_data[y][x] = pd.Series(t_dp["E" + str(x + 1)]).reset_index(drop=True)
        
        new_data[y][w-3] = t_dp["label"].values[0]
        new_data[y][w-2] = t_dp["participant"].values[0]
        new_data[y][w-1] = t_dp["trial_id"].values[0]
        

    columns = ["E%d" % i for i in range(1, 61)] + ["label", "participant", "trial_id"]
    # make a new dataframe with the new data
    new_df = pd.DataFrame(data=new_data, columns=columns)
    return new_df