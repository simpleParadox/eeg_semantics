print("start imports", flush=True)

import run_setup

import pandas as pd
import numpy as np

from jwlab.ml_prep import load_ml_df, y_to_binary
from sklearn.model_selection import cross_val_score
from jwlab.eval import eval_normal
from sktime.pipeline import Pipeline
from sktime.transformers.compose import ColumnConcatenator
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.classifiers.dictionary_based.boss import BOSSEnsemble
from sktime.classifiers.compose import ColumnEnsembleClassifier
from sklearn.model_selection import train_test_split
import sys

print("finish imports, start loading df", flush=True)

df = load_ml_df(sys.argv[1])
print("loaded_df", flush=True)

y = y_to_binary(df.label.values.flatten())
df = df.drop(columns=["label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

model = ColumnEnsembleClassifier(estimators=[
    ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
    ("BOSSEnsemble3", BOSSEnsemble(ensemble_size=3), [3]),
])

model.fit(X_train, y_train)
print(np.mean(model.predict(X_test) != y_test))