#!/bin/bash
#SBATCH --partition=modi_short
#SBATCH --exclusive
#SBATCH -o ./results/slurm_sequential_%j.out # STDOUT
num_voxels=4
echo num_voxels=$num_voxels
echo sequential
# Run the program
singularity exec ~/modi_images/hpc-notebook_latest.sif ./ct_sequential \
--num-voxels $num_voxels --input ~/modi_mount/CT_Reconstruction/ct_data
# ~/modi_mount/CT_Reconstruction/ct_data
# ~/modi_readonly/ct_data

