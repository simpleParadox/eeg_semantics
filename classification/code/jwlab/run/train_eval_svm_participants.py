print("start imports", flush=True)

import run_setup

import pandas as pd
import numpy as np

from jwlab.ml_prep import load_ml_df, y_to_binary, no_average, average_trials
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from jwlab.eval import eval_across_categories
import sys

print("finish imports, start loading df", flush=True)

df = load_ml_df(sys.argv[1])
print("loaded_df", flush=True)

X,y,w,p = no_average(df)
print("prepped train_data", flush=True)

y = y_to_binary(y)

model = LinearSVC()
print("model created", flush=True)

errs = eval_across_categories(model, X, y, p, 200, random_state=50)
print("\n---\n")
for j in range(errs.shape[0]):
  print("(Participant %d) Accuracy: %0.2f (+/- %0.2f)" % (p[j], errs[j, :].mean(), errs[j, :].std() * 2))