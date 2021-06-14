#!/bin/bash


for i in 4 
do
	for j in 0.00001 0.00002
	do
		sbatch --job-name=bioel_5.$i.$j --output=bioel_5.$i.$j.out biomed_el_job.slurm 'pretraining5' $i $j 5
	done
done


