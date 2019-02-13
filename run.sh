#!/bin/bash
#SBATCH --mem=600m
#SBATCH -c4
#SBATCH --time=1-0
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=gal.patel@mail.huji.ac.il

dir=/cs/labs/oabend/gal.patel/LabProject/scaffold/

cd $dir

source /cs/labs/oabend/gal.patel/virtualenvs/scaffold-venv/bin/activate
python -m allennlp.run evaluate \
    --archive-file model_np_pp_scaf_fsp.tar.gz \
    --evaluation-data-file data/fndata-1.5/test/fulltext/ \
    --cuda-device 0
