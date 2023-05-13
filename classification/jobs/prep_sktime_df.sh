#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=output/sktimeprep.out

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install sklearn
pip install sktime==0.3.0
pip install tsfresh
pip install --no-index -r general_requirements.txt
python /$HOME/projects/def-jwerker/kjslakov/jwlab_eeg/classification/code/jwlab/run/prep_sktime_df.py /$HOME/projects/def-jwerker/kjslakov/data/ml_df_sktime.pkl