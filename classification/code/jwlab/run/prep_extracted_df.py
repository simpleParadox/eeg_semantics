import run_setup

import pandas as pd
import numpy as np

from jwlab.run.computecanada_constants import data_path, bad_trials_path, df_path
from jwlab.ml_prep import create_ml_df_internal, load_ml_data, save_ml_df
from jwlab.bad_trials import get_bad_trials, transform_ybad_indices
from jwlab.first_participants_map import map_first_participants
from tsfresh import extract_relevant_features
import sys

participants = ["107", "109", "111", "112", "115", "116", "904", "905", "906", "908", "909", "910", "912"]

df, ys = load_ml_data(data_path, participants)

print("loaded data", flush=True)

df = df[df.Time >= 0]
df = df.drop(columns=["E65", "E64", "E63", "E62", "E61"], axis=1)

df['id'] = np.concatenate([[i] * 1000 for i in range(len(df.index) // 1000)])

ybad = get_bad_trials(participants, ys, bad_trials_path)
ys = map_first_participants(ys, participants)
y = np.concatenate(ys)
ybad = transform_ybad_indices(ybad, ys)
y[ybad] = -1

y = pd.Series(y)

features_filtered_direct = extract_relevant_features(df, y, column_id='id', column_sort='Time')

save_ml_df(features_filtered_direct, sys.argv[1])

