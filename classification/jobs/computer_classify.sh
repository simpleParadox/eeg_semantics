#!/bin/bash
#SBATCH --account=def-jwerker
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0:01:00

module load python/3.6
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r computer_classify_requirements.txt
python ~/projects/def-jweker/kjslakov/jwlab_eeg/classification/computers_classify.py