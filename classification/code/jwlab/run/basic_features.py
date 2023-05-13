print("start imports", flush=True)

import run_setup

import pandas as pd
import numpy as np

from jwlab.ml_prep import load_ml_df, y_to_binary
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import sys

print("finish imports, start loading df", flush=True)

df = load_ml_df(sys.argv[1])
print("loaded_df", flush=True)

y = y_to_binary(df.label.values.flatten())
df = df.drop(columns=["label"], axis=1)

model = LinearSVC()

scores = cross_val_score(model, X, y, cv=5)

print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))