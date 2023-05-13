#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --mem=6G
#SBATCH --time=0:10:00

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install sklearn --no-index
pip install --no-index -r general_requirements.txt
python /$HOME/projects/def-jwerker/kjslakov/jwlab_eeg/classification/code/jwlab/run/prep_ml_df.py