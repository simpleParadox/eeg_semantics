#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --mem=8G
#SBATCH --time=12:40:00
#SBATCH --output=output/%x-%j.out

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index sklearn
pip install --no-index -r general_requirements.txt
mkdir $SLURM_TMPDIR/data
echo "copying dataset"
cp /$HOME/projects/def-jwerker/kjslakov/data/ml_df_readys.pkl $SLURM_TMPDIR/data/dataset.pkl
echo "dataset copied"
python /$HOME/projects/def-jwerker/kjslakov/jwlab_eeg/classification/code/jwlab/run/train_eval_svm_participants.py $SLURM_TMPDIR/data/dataset.pkl