#!/bin/bash -l
#SBATCH --job-name=EdgeVolution
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail 
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python
module load cuda
module load cudnn

conda activate edgevolution

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORK/software/private/conda/envs/edgevolution/lib

srun python main.py +hyperparameters=speech_commands +search_space=speech_commands +boards=none
# srun python main.py +hyperparameters=emg_airob +search_space=emg_airob +boards=none
# srun python main.py +hyperparameters=cifar10 +search_space=cifar10 +boards=none
# srun python main.py +hyperparameters=daliac +search_space=daliac +boards=none
