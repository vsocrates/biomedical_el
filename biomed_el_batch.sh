#!/bin/bash

for i in 4 8 16
do
	for j in  0.00002 0.0001
	do
		for stringval in pretraining pretraining2 pretraining3 pretraining4
		do
			sbatch --job-name=bioel_$stringval.$i.$j --output=bioel_$stringval.$i.$j.out biomed_el_job.slurm $stringval  $i $j 25
		done
	done
done


