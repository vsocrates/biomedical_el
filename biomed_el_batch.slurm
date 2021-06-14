#!/bin/bash

#SBATCH --job-name=train_luke_medmention
#SBATCH --output=train_luke_medmen-%j.out

#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100G
#SBATCH --nodes=1
#SBATCH --partition=general,interactive,gpu
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx500

#SBATCH --time=0-04:00:00

module restore fosscuda111

# using your anaconda environment
source activate biomed_el111

python train.py --input-data-file /home/vs428/project/MedMentions/full/data/corpus_pubtator.txt \
		--train-pmids /home/vs428/project/MedMentions/full/data/corpus_pubtator_pmids_trng.txt \ 
		--test-pmids /home/vs428/project/MedMentions/full/data/corpus_pubtator_pmids_test.txt \
		--entity-vocab  /home/vs428/project/MedMentions/full/pretraining5/entity_vocab.jsonl \ 
		--luke-model /home/vs428/project/MedMentions/full/pretraining5/model_epoch20.bin \
		--luke-metadata /home/vs428/project/MedMentions/full/pretraining5/metadata.json \ 
		 --cuda --batch-size 


