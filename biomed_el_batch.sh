#!/bin/bash

batchsize=( 4 8  )
warmups=( 8000 6000 )

for i in "${!batchsize[@]}"
do
	for schedule in linear cosine warmup
	do
		sbatch --job-name=bioel_pretraining4.${batchsize[$i]}.$schedule --output=bioel_pretraining4.${batchsize[$i]}.$schedule.out biomed_el_job.slurm pretraining4 ${batchsize[$i]}  0.0001  50 $schedule ${warmups[$i]}
	done 
done


