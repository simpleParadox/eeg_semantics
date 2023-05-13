from ml_prep import load_ml_data
import numpy as np
import pandas as pd
from bad_trials import get_bad_trials, transform_ybad_indices
from first_participants_map import map_first_participants
from tsfresh import extract_relevant_features

def prep_tsfresh_data(df, ys, participants):
    df = df[df.Time >= 0]
    df = df.drop(columns=["E65", "E64", "E63", "E62", "E61"], axis=1)

    df['id'] = np.concatenate([[i] * 1000 for i in range(len(df.index) // 1000)])

    ybad = get_bad_trials(participants, ys)
    ys = map_first_participants(ys, participants)
    y = np.concatenate(ys)
    ybad = transform_ybad_indices(ybad, ys)
    y[ybad] = -1

    y = pd.Series(y)

    features_filtered_direct = extract_relevant_features(df, y,
                                                     column_id='id', column_sort='Time')

    return df, features_filtered_direct

