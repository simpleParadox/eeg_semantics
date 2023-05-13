import run_setup

import pandas as pd
import numpy as np

from jwlab.run.computecanada_constants import data_path, bad_trials_path, df_path
from jwlab.ml_prep import create_ml_df_internal, load_ml_data, save_ml_df


participants = ["107", "109", "111", "112", "115", "116", "904", "905", "906", "908", "909", "910", "912"]

df, ys = load_ml_data(data_path, participants)
df_final = create_ml_df_internal(df, ys, participants, downsample_num=1000, bad_trials_filepath=bad_trials_path)
save_ml_df(df_final, df_path)