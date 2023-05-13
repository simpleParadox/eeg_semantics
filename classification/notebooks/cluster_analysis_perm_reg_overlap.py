#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import setup_jwlab

import sys
import argparse
seed = 1# int(sys.argv[1])
# print("SLURM array task ID: ", sys.argv[1])

sys.path.insert(1, 'classification/code')
from jwlab.constants import cleaned_data_filepath
from jwlab.cluster_analysis_perm import cluster_analysis_procedure
from jwlab.ml_prep_perm import prep_ml, slide_df, init, load_ml_data, get_bad_trials, map_participants,average_trials_and_participants
from jwlab.bad_trials import get_bad_trials, get_left_trial_each_word


from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, RepeatedKFold
from scipy import stats
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


# In[3]:


# Argument 1: 9 or 11 (month olds)
# Argument 2: Boolean, True to randomize the labels, False otherwise
# Argument 3: averaging, could be: no_averaging, average_trials, average_trials_and_participants, permutation
# Argument 4: sliding_window_config[start_time, end_time, window_lengths[], step_length]
# Argument 5: cross_val_config[num_fold, num_fold iterations, number of sample iterations]

#cluster_analysis_procedure(9, False, "permutation", [-200, 1000, [10, 20, 40, 60], 10], [3, 5])


# In[ ]:


# Initialize argparse.
parser = argparse.ArgumentParser(description='Run decoding analysis on EEG data.')
parser.add_argument('--age_group', type=int, help='Age of participants. 9 or 12.', required=True, default=9)
parser.add_argument('--randomize_labels', type=bool, help='Whether to randomize the labels for the permutation test.', required=True, default=False)
parser.add_argument('--animacy', type=bool, help='Whether to do animacy decoding.', required=True, default=False)
parser.add_argument('--tgm', type=bool, help='Whether to do TGM analysis.', required=True, default=False)
parser.add_argument('--across_groups', type=bool, help='Whether to do across groups analysis.', required=True, default=False)
parser.add_argument('--across_groups_group_1', type=int, help='Age of participants for group 1. 9 or 12.', required=False, default=9)
parser.add_argument('--across_groups_group_2', type=int, help='Age of participants for group 2. 9 or 12.', required=False, default=12)
parser.add_argument('--embeddings', type=str, help='Which embeddings to use. Options are w2v, fine_tuned, or phoneme', required=True, default='w2v')
parser.add_argument('--monte_carlo_iterations', type=int, help='Number of Monte Carlo iterations to run.', required=False, default=50)

args = parser.parse_args()
averaging = 'average_trials_and_participants'  # For regular decoding.
if args.tgm:
    averaging = 'tgm'
elif args.across_groups:
    averaging = 'across'

permutation = False
if args.randomize_labels:
    permutation = True

age_group = 9
if args.age_group == 12:
    age_group = 12

if args.age_group not in [9, 12]:
    raise ValueError('Age group must be 9 or 12.')


if args.embeddings not in ['w2v', 'fine_tuned', 'phoneme']:
    raise ValueError('Embeddings must be w2v, fine_tuned, or phoneme. Please choose one of these options.')





# NOTE: If you set useRandomizedLabel = True and set type='simple', it will run the null_distribution. But you have to run it 100 times/jobs.
# result = cluster_analysis_procedure(12, False, "average_trials_and_participants", [-200, 1000, [100], 10], [5, 4, 50], type='simple', animacy=False, no_animacy_avg=False, do_eeg_pca=False, do_sliding_window=False)
result = cluster_analysis_procedure(age_group, permutation, averaging, [-200, 1000, [100], 10], [5, 4, args.monte_carlo_iterations], type='simple', seed=seed, animacy=args.animacy, no_animacy_avg=False, do_eeg_pca=False, do_sliding_window=False, embeddings=args.embeddings, age_group_1=args.across_groups_group_1, age_group_2=args.across_groups_group_2)
