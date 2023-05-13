import pandas as pd
import numpy as np
import setup_jwlab
from jwlab.constants import cleaned_data_filepath
from jwlab.cluster_analysis_perm import cluster_analysis_procedure, prep_ml, cross_validaton
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, RepeatedKFold
from scipy import stats
import more_itertools as mit
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

results = cluster_analysis_procedure(11, False, "permutation", [-200, 1000, [10], 10], [3, 15, 20])