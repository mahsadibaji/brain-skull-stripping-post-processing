#!/bin/sh

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node=1

module load python/anaconda3-2018.12

source activate /work/souza_lab/mahsa/software/miniconda3/envs/torchenv

python -V
conda info --env
conda list

python /work/souza_lab/mahsa/U-Net-Segmentation/Scripts/run_train.py --batch_size 8 --learning_rate 0.001 --epochs 100 --val_split 0.3 --results_dir /work/souza_lab/mahsa/U-Net-Segmentation/train_results_3/ --source_dev_images /work/souza_lab/mahsa/U-Net-Segmentation/source_train_set.csv --source_dev_masks /work/souza_lab/mahsa/U-Net-Segmentation/source_train_set_masks.csv --verbose True


conda deactivate
