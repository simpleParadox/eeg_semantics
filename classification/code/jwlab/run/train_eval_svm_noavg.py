print("start imports", flush=True)

import run_setup

import pandas as pd
import numpy as np

from jwlab.ml_prep import load_ml_df, y_to_binary, no_average, average_trials
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from jwlab.eval import eval_normal
import sys

print("finish imports, start loading df", flush=True)

df = load_ml_df(sys.argv[1])
print("loaded_df", flush=True)

X,y,w,p = no_average(df)
print("prepped train_data", flush=True)

y = y_to_binary(y)

model = LinearSVC()
print("model created", flush=True)

scores = cross_val_score(model, X, y, cv=5)

print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))