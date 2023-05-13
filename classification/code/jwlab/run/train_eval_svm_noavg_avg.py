import run_setup

import pandas as pd
import numpy as np

from jwlab.run.computecanada_constants import df_path
from jwlab.ml_prep import load_ml_df, y_to_binary, no_average, average_trials
from sklearn.svm import LinearSVC

df = load_ml_df(df_path)
df_2 = df.copy()

X,y,w,p = no_average(df)
X_2,y_2,w_2,p_2 = average_trials(df_2)


model = LinearSVC(C=1.0, max_iter=1000)


num_trials = 10
errs = np.zeros(num_trials)
for i in range(num_trials):
    X_train, y_train, X_test, y_test = X[np.random.randint(X.shape[0], size=X.shape[0]), :], np.random.choice(y, y.shape[0]), X_2[np.random.randint(X_2.shape[0], size=X_2.shape[0]), :], np.random.choice(X_2, X_2.shape[0])
    model.fit(X_train, y_train)
    
    errs[i] = np.mean(model.predict(X_test) != y_test)
    sys.stdout.write('\r')
    percent = (i + 1) / num_trials
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*percent), percent*100))


print(errs)
print(np.mean(errs))