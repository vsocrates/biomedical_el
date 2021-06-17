#!/bin/bash


#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100G
#SBATCH --nodes=1
#SBATCH --partition=general,interactive,gpu
#SBATCH --gres=gpu:1

#SBATCH --time=0-06:00:00

PRETRAIN_NUM=$1
BATCH_SIZE=$2 
LR=$3
NUM_EPOCHS=$4
SCHEDULER=$5 
WARMUP_STEPS=$6 

module restore fosscuda111

# using your anaconda environment
source activate biomed_el111

echo "/home/vs428/project/MedMentions/full/$PRETRAIN_NUM/entity_vocab.jsonl"

python train.py --input-data-file "/home/vs428/project/MedMentions/full/data/corpus_pubtator.txt" \
		--train-pmids "/home/vs428/project/MedMentions/full/data/corpus_pubtator_pmids_trng.txt" \
		--test-pmids "/home/vs428/project/MedMentions/full/data/corpus_pubtator_pmids_test.txt" \
		--entity-vocab "/home/vs428/project/MedMentions/full/$PRETRAIN_NUM/entity_vocab.jsonl" \
		--luke-model "/home/vs428/project/MedMentions/full/$PRETRAIN_NUM/model_epoch20.bin" \
		--luke-metadata "/home/vs428/project/MedMentions/full/$PRETRAIN_NUM/metadata.json" \
		--cuda --batch-size $BATCH_SIZE --lr $LR --num-epochs $NUM_EPOCHS --lr-scheduler $SCHEDULER \
		--num-warmup-steps $WARMUP_STEPS 
