from sklearn.model_selection import train_test_split
import sys
import numpy as np

def eval_normal(model, X, y, num_trials, test_size=0.2, random_state=0, progress_bar=False):
    errs = np.zeros(num_trials)
    for i in range(num_trials):
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=test_size, random_state=random_state+i)
        model.fit(X_train, y_train)
        
        errs[i] = np.mean(model.predict(X_test) != y_test)
        if progress_bar:
            sys.stdout.write('\r')
            percent = (i + 1) / num_trials
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*percent), percent*100))
    return errs

def eval_across_categories(model, X, y, cat, num_trials, test_size=0.2, random_state=0):
    num_categories = int(np.max(cat)) + 1
    errs = np.zeros((num_categories, num_trials))
    for i in range(num_trials):
        X_train, X_test, y_train, y_test, _, cat_test = train_test_split(X, y, cat, test_size=test_size, random_state=i+random_state)
        model.fit(X_train, y_train)
        
        for j in range(num_categories):
            if len(y_test[cat_test == j]) == 0:
                errs[j, i] = 0.5
            else:
                errs[j, i] = np.mean(model.predict(X_test[cat_test == j]) != y_test[cat_test == j])
        
        sys.stdout.write('\r')
        percent = (i + 1) / num_trials
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*percent), percent*100))
    return errs