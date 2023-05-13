import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('G:\jw_lab\jwlab_eeg\classification\code')
from matplotlib.gridspec import GridSpec
from jwlab.data_graph import plot_good_trial_participant, plot_good_trial_word
from jwlab.participants_map import map_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word
from jwlab.constants import word_list, bad_trials_filepath, old_participants, cleaned_data_filepath, cleaned_data_filepath_new_filter

from meegkit.detrend import regress, detrend, create_masked_weight
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_ml_data(detrend_bool = False):
    # participants = ["904", "905", "906", "908", "909", "910", "912", "913", "914", "916", "917", "919", "920", "921",
    #                 "923", "924", "927", "929", "928", "930", "932"]

    # all
    #participants = [ "904", "905","906", "908", "909", "912", "913", "914", "916", "917", "919", "920", "921", "923", "924", "927", "929","928", "930", "932"]
    participants = ["105"]#, "106", "107", "109", "111", "112", "115", "116", "117", "119", "120", "121", "122", "124"]
    # read all participant csvs, concat them into one dataframe
    if participants[0][0] == '1':
        dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (cleaned_data_filepath_new_filter, s)) for s in participants]
    else:
        dfs = [pd.read_csv("%s%s_cleaned_ml.csv" % (cleaned_data_filepath_new_filter, s)) for s in participants]

    df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
    df = df.drop('E65', axis=1)
    jenn_local_label_filepath = "G:\jw_lab\jwlab_eeg\Data\Imported\label_jennlocal\\"

    # ys = [np.loadtxt("%s%s_labels.txt" % (cleaned_data_filepath, s)).tolist()
    #       for s in participants]

    ys = [np.loadtxt("%s%s_labels.txt" % (jenn_local_label_filepath, s)).tolist()
          for s in participants]

    detrending = detrend_bool
    if detrending == False:
        # Write code here for the scaling operation only.
        print("No detrending")
        # scaler = StandardScaler()
        # scaled_df = scaler.fit_transform(df.iloc[:, :-1].values)  # Excluding the 'Time' column.
        # new_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns[:-1])
        # df = pd.concat([new_df, df['Time']], axis=1)

        return df, ys
    else:
        # 'detrending = True.
        # The following code section is for masked-trial robust detrending.
        # ----------------------------------------
        # Change parameters here.
        scaling_first = True
        pre_onset = False  # doesn't really matter when masked_trial_detrending is 'False'.
        masked_trial_detrending = True  # Set this to 'False' for regular robust detrending.

        # -------------------------------------------------------------------
        if scaling_first == True:
            # scaler = StandardScaler()
            # scaled_df = scaler.fit_transform(df.iloc[:, :-1].values)
            new_df = df.iloc[:,:-1] #pd.DataFrame(scaled_df, index=df.index, columns=df.columns[:-1])

            if pre_onset == False:
                # Setting post-onset weights to zero.
                print('post-onset')
                events = np.arange(0, len(new_df), 1200)  # np.arange(0, len(new_df), 1200)
                tmin = 0.2  # 0
                tmax = 1.2  # 0.2
            else:
                # Setting pre-onset weights to zero
                print('pre-onset')
                events = np.arange(0, len(new_df), 1200)
                tmin = 0
                tmax = 0.2
            sfreq = 1000

            detrend_weights = None
            if masked_trial_detrending == False:
                print('Regular Robust detrending')
                y, w, r = detrend(new_df.to_numpy(), order=2, w=None)
            else:
                print('Masked Trial Robust detrending')
                print('No scaling')
                detrend_weights = create_masked_weight(new_df, events, tmin, tmax, sfreq)
                y, w, r = detrend(new_df.to_numpy(), order=30, w=detrend_weights)

            # Add the 'Time' column back.
            df_with_time = pd.concat([pd.DataFrame(y, index=df.index, columns=df.columns[:-1]), df['Time']], axis=1)
            return df_with_time, ys
        else:
            df_no_time = df.drop('Time', axis=1)
            if pre_onset == False:
                # Setting post-onset weights to zero.
                events = np.arange(0, len(df_no_time), 1200)  # np.arange(0, len(new_df), 1200)
                tmin = 0.2  # 0
                tmax = 1.2  # 0.2
            else:
                # Setting pre-onset weights to zero
                events = np.arange(0, len(df_no_time), 1200)
                tmin = 0
                tmax = 0.2
            sfreq = 1000

            detrend_weights = None
            if masked_trial_detrending == False:
                print('Regular robust detrending')
                y, w, r = detrend(df_no_time.to_numpy(), order=2, w=None)
            else:
                print('Masked-trial robust detrending')
                detrend_weights = create_masked_weight(df_no_time, events, tmin, tmax, sfreq)
                y, w, r = detrend(df_no_time.to_numpy(), order=1, w=detrend_weights)
            scaler = StandardScaler()
            scaled_df = scaler.fit_transform(y)  # .iloc[:, :-1].values)
            new_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns[:-1])
            df_with_time = pd.concat([new_df, df['Time']], axis=1)
            return df_with_time, ys


df, ys = load_ml_data(detrend_bool=False)


# Average the trials for the time window from 0-1000 in df.
def avg_trials(df):
    # First select the data from 0-1000ms
    data = df
    pre_window = 200
    event_start = np.arange(0, len(df), 1200)
    selected_data = []
    for onset in event_start:
        start = pre_window + onset
        end = onset + pre_window + 1000
        selected_data.append(data.iloc[start:end, :-1].values)

    selected_data = np.array(selected_data)
    temp = selected_data.mean(axis=0)

    return temp

df = avg_trials(df)
plt.clf()
plt.ylim(-150, 150)
plt.plot(np.arange(-200,1000), df[0:1200, 50:60])#.iloc[200:1200,:-1].values)
plt.title('Without base_correct_no_detrending')
plt.show()