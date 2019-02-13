#!/bin/bash
#SBATCH --mem=6000m
#SBATCH -c4
#SBATCH --time=7-0
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=gal.patel@mail.huji.ac.il

dir=/cs/labs/oabend/gal.patel/tasks/RevolvingDoors/

cd $dir

source /cs/labs/oabend/gal.patel/virtualenvs/haystack-venv/bin/activate
python entities_organization.py
