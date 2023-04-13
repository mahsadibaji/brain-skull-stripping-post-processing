#!/bin/sh

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node=1

module load python/anaconda3-2018.12

source activate /work/souza_lab/mahsa/software/miniconda3/envs/torchenv

python -V
conda info --env

python /work/souza_lab/mahsa/U-Net-Segmentation/Scripts/run_test_LPBA40.py --results_dir /work/souza_lab/mahsa/U-Net-Segmentation/LPBA40-results/ --source_test_images /work/souza_lab/mahsa/LPBA40/LPBA40_subjects_list.csv --source_test_masks /work/souza_lab/mahsa/LPBA40/LPBA40_masks_list.csv --model_path /work/souza_lab/mahsa/U-Net-Segmentation/train_results_3/unet.pth --verbose True


conda deactivate