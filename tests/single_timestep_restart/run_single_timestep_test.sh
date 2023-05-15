#!/bin/bash

# Minimal runscript for single timestep
ulimit -s unlimited

OPTIND=1

# MiMA must be compiled and all modules must be loaded before running
N_PROCS=4
PATH_TO_CCOMB="../../build/mppnccombine"
PATH_TO_EXEC="../../build/mima.x"
TOLERANCE=1e-8

# OPTIONAL VALUES TO CHANGE PATH TO EXECUTABLE ETC
while getopts "p:x:c:t:" flag
do
    case "${flag}" in
        p) N_PROCS=${OPTARG};;
        x) PATH_TO_EXEC=${OPTARG};;
        c) PATH_TO_CCOMB=${OPTARG};;
        t) TOLERANCE=${OPTARG};;
    esac
done

echo "Number of processors: $N_PROCS"
echo "Path to mima executable: $PATH_TO_EXEC"
echo "Path to mppncombine executable: $PATH_TO_CCOMB"

ulimit -s unlimited

echo "run MiMA"
[ ! -d RESTART ] && mkdir RESTART

srun --ntasks $N_PROCS $PATH_TO_EXEC

echo "Run complete. Postprocess with mppncombine in ${PATH_TO_CCOMB}"

${PATH_TO_CCOMB} -r atmos_daily.nc atmos_daily.nc.????
${PATH_TO_CCOMB} -r atmos_avg.nc atmos_avg.nc.????

echo "Done"

echo "Test is the output the same as the baseline files, considering tolerance $TOLERANCE ?"
cdfdiff -t $TOLERANCE,0 atmos_daily.nc  atmos_daily_base.nc 
cdfdiff -t $TOLERANCE,0 atmos_avg.nc  atmos_avg_base.nc


