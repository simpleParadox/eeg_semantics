#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --mem=6G
#SBATCH --time=0:10:00

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install sklearn
pip install sktime
pip install --no-index -r general_requirements.txt
python /$HOME/campbejc/projects/def-campbejc/campbejc/jwlab_eeg/classification/code/jwlab/run/prep_ml.py