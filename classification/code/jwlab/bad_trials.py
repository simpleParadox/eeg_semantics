import numpy as np
import pandas as pd
from math import isnan
# <<<<<<< original
from jwlab.constants import bad_trials_filepath, db_filepath, cleaned_data_filepath
# =======
# from jwlab.constants import bad_trials_filepath, db_filepath, cleaned_data_filepath, db_abs_200uv_filepath
# db_filepath = db_abs_200uv_filepath  #NOTE: Comment this line if using the old db_filepath.
# >>>>>>> main
print(bad_trials_filepath)
bad_trial_df = pd.read_csv(bad_trials_filepath)
bad_trial_df.Ps = bad_trial_df.Ps.interpolate(method="pad")
# drop "looking left" trials because they are not considered as bad trials
bad_trial_df = bad_trial_df[bad_trial_df['Reason'] != "left"]
bad_trial_df = bad_trial_df.drop_duplicates(subset=["Ps", "Cell", "Observation"])

def get_bad_trials(participants):
    # Appending bad trials either from cell&obs columns (new segs) or tIndex columns (old segs)
    ybad = []
    # loop through all participants
    for i in range(len(participants)):
        # generate a dataframe for each participants
        p_df = bad_trial_df[bad_trial_df.Ps == int(participants[i])]
        if len(p_df) == 0:  # if this participant's trials are all good (i.e. no bad trials)
            ybad.append([])  # append empty list
        else:  # append bad trials from table (deprecated: tIndex)
            # Retrieve bad trials based on the cell and obs columns
            ybad.append(get_ybad_from_cel_obs(participants, i, bad_trial_df))

    # convert all bad trial indices to int
    ybad = [[int(y) for y in x] for x in ybad]

    return ybad


def get_ybad_from_cel_obs(participants, i, df):
    bad_trials = []
    orig_trial_df = pd.read_csv(
        "%s%s_trial_cell_obs.csv" % (db_filepath, participants[i]))
    for index, row in df.iterrows():
        # get the trial index by searching for matched combination of cell, obs and participant value
        bad_trials += orig_trial_df[(orig_trial_df['cell'] == row['Cell']) & (orig_trial_df['obs']
                                                                              == row['Observation']) & (int(participants[i]) == row['Ps'])].trial_index.values.tolist()
    return bad_trials


def get_left_trial_each_word(participants):
    rt = []
    for participant in participants:
        orig_word_count_df = pd.read_csv(
            "%s%s_trial_cell_obs.csv" % (db_filepath, participant))
        orig_word_count = orig_word_count_df.groupby(['cell']).size()
        
        bad_word_count_df = bad_trial_df[bad_trial_df.Ps == int(participant)]
        bad_word_count = bad_word_count_df.groupby(['Cell']).size()
        
        rt += [orig_word_count.subtract(bad_word_count,
                                        fill_value=0).astype(int)]
        
    return rt
