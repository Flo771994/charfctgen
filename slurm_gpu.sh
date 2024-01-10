#!/bin/sh
#SBATCH --job-name 10dimrun            # this is a parameter to help you sort your job when listing it
#SBATCH --error protocols/testrun-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output protocols/testrun-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --gpus=1
#  #SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu (seems to be overwrittem when less than default)
#SBATCH --mem=10000             #memory needed (3GB default)
#SBATCH --time 60:00                  # maximum run time.
 

module purge                           #Unload all modules


module load Anaconda3              # load Anaconda
module load CUDA/11.8.0             # load CUDA for GPU computing
## define and create a unique scratch directory
#SCRATCH_DIRECTORY=/global/work/${USER}/kelp/${SLURM_JOBID}
#mkdir -p ${SCRATCH_DIRECTORY}
#cd ${SCRATCH_DIRECTORY}

echo "before calling source: $PATH"
echo "Home: $HOME"
# Activate Anaconda work environment
# source $HOME/.bashrc
source activate charfctgpu # conda activate charfct does not work

echo "after calling source: $PATH"


srun python main.py                      # run your software