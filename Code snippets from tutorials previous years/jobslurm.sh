#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=00:05:00
#SBATCH --partition=dev_multiple
#SBATCH --ntasks-per-node=25
#SBATCH --mail-user=<my_email_address>
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
echo "Loading Pythona module and mpi module"
module load devel/python/3.9.2_gnu_10.2
module load compiler/gnu/12.1
module load mpi/openmpi/4.1
module list
startexe="mpirun --bind-to core --map-by core -report-bindings python3 ./ProcessorTest.py"
exec $startexe
