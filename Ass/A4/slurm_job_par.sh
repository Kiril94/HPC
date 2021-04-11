#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=modi_short
#SBATCH --exclude=modi000
#SBATCH -o ./results/slurm_parallel_%j.out # STDOUT
# Set number of OpenMP thread to use
# set loop scheduling to static
export OMP_SCHEDULE=static
# Schedule one thread per core. Change to "threads" for hyperthreading
export OMP_PLACES=cores
# Place threads as close to each other as possible
export OMP_PROC_BIND=close
# export OMP_DISPLAY_ENV=true
# Set number of OpenMP thread to use (this should be 64 cores / number of ranks pr node)
export OMP_NUM_THREADS=1
num_voxels=128
echo Nvoxels=$num_voxels
echo Nthreads=$OMP_NUM_THREADS
# Run the program
mpirun --mca btl_openib_warn_no_device_params_found 0\
        singularity exec ~/modi_images/hpc-notebook_latest.sif ./ct_parallel02 \
        --num-voxels $num_voxels --input ~/modi_readonly/ct_data
# ~/modi_readonly/ct_data
#  -N $NNODES \
#       -n $NPROCS \
# ~/modi_mount/CT_Reconstruction/ct_data

#  #SBATCH --exclude=modi000