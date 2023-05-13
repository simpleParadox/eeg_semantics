# ------------ WORD LIST ------------

word_list = ["baby", "BAD_STRING", "bird", "BAD_STRING", "cat", "dog", "duck", "mommy",
             "banana", "bottle", "cookie", "cracker", "BAD_STRING", "juice", "milk", "BAD_STRING"]


# ------------ OLD PARTICIPANTS ------------
#old_participants = ["107", "904", "905", "906"]
old_participants = []

# ------------ FILE PATH ------------
cleaned_data_filepath = ''
bad_trials_filepath = ''
db_filepath = ''
df_filepath = ''
df_filepath_sktime = ''

import sys
sys.path.insert(1, 'classification/code')
from jwlab.profile import user
print(user)
if user == 'user':
    # ---- File path on Local machine.
    cleaned_data_filepath = "data/cleaned2/"
    bad_trials_filepath = "data/ML_badtrials-Table 2.csv"
    db_filepath = "data/db2/"

# elif user == 'usercc':
#     # File path on remote server.
#     cleaned_data_filepath = "data/cleaned2/"
#     bad_trials_filepath = "data/ML_badtrials-Table 2.csv"
#     db_filepath = "data/db2/"
