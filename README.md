## Installing packages.
The following python packages are required to run.
Anaconda environments are supported.
- numpy
- scikit-learn
- matplotlib
- pandas
- seaborn
- scipy
- setuptools
- more_itertools

As of now, only Linux and macOS are supported.

Install any packages that may be missing.


## Data
Download the 'data' folder inside 'preprocessed_data' folder from osf and store in the root directory of the project.
Link: https://osf.io/hca7b/

## Running the code.
The starting point of the code is the classification/cluster_analysis_perm_reg_overlap.py file.
```
python classification/notebooks/cluster_analysis_perm_reg_overlap.py -h
usage: cluster_analysis_perm_reg_overlap.py [-h] --age_group AGE_GROUP --randomize_labels RANDOMIZE_LABELS --animacy ANIMACY --tgm TGM --across_groups ACROSS_GROUPS
                                            [--across_groups_group_1 ACROSS_GROUPS_GROUP_1] [--across_groups_group_2 ACROSS_GROUPS_GROUP_2] --embeddings EMBEDDINGS
                                            [--monte_carlo_iterations MONTE_CARLO_ITERATIONS]

Run decoding analysis on EEG data.

optional arguments:
  -h, --help            show this help message and exit
  --age_group AGE_GROUP
                        Age of participants. 9 or 12.
  --randomize_labels RANDOMIZE_LABELS
                        Whether to randomize the labels for the permutation test.
  --animacy ANIMACY     Whether to do animacy decoding.
  --tgm TGM             Whether to do TGM analysis.
  --across_groups ACROSS_GROUPS
                        Whether to do across groups analysis.
  --across_groups_group_1 ACROSS_GROUPS_GROUP_1
                        Age of participants for group 1. 9 or 12.
  --across_groups_group_2 ACROSS_GROUPS_GROUP_2
                        Age of participants for group 2. 9 or 12.
  --embeddings EMBEDDINGS
                        Which embeddings to use. Options are w2v, fine_tuned, or phoneme
  --monte_carlo_iterations MONTE_CARLO_ITERATIONS
                        Number of Monte Carlo iterations to run.
```
NOTE: If you face an error while running the code, please open an issue on GitHub, or start a discussion thread.