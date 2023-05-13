#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=3-01:30:00
#SBATCH --output=output/extracting.out

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install sklearn
pip install sktime
pip install tsfresh
pip install --no-index -r general_requirements.txt
python /$HOME/projects/def-jwerker/kjslakov/jwlab_eeg/classification/code/jwlab/run/prep_extracted_df.py /$HOME/projects/def-jwerker/kjslakov/data/ml_df_extracted.pkl