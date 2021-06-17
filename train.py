import json
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F 
from torch import autograd

import numpy as np 
import time
import math 


from entitylinker.model_utils import *
from entitylinker.train_utils import *

from entitylinker.dataset import MedMentionsDataset, Collater
from entitylinker.model import EntityLinker

import transformers
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_constant_schedule_with_warmup,get_constant_schedule

import argparse

parser = argparse.ArgumentParser(description='Train the Biomedical Entity Linking Model.')
parser.add_argument('--input-data-file', help='full MedMentions data file')
parser.add_argument('--train-pmids', help='train set pmids')
parser.add_argument('--test-pmids', help='test set pmids')
parser.add_argument('--entity-vocab', help='entity vocab file')
parser.add_argument('--luke-model', help='LUKE entity embedding model')
parser.add_argument('--luke-metadata', help='LUKE entity embedding metadata')

parser.add_argument('--bert-model-name', help="name of bert model", default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
parser.add_argument('--batch-size', help="batch size of training", default=16, type=int)
parser.add_argument('--lr', help="learning rate", default=0.00002, type=float)
parser.add_argument('--max-length', help="max input length of BERT model", default=128, type=int)
parser.add_argument('--num-epochs', help="number of epochs to train", default=20, type=int)
parser.add_argument("--lr-scheduler", help="Learning rate scheduler", choices=['linear', 'cosine', 'warmup', 'constant'])
parser.add_argument("--num-warmup-steps", help="number of warmup steps for scheduler", default=50)
parser.add_argument('--cuda', action='store_true', help='use cuda?')

args = parser.parse_args()



# Import the Device
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu") 

if torch.cuda.is_available():
    
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


# Input file params
# input_data_file = "/home/vs428/project/MedMentions/full/data/corpus_pubtator.txt"
# train_pmids_file = "/home/vs428/project/MedMentions/full/data/corpus_pubtator_pmids_trng.txt"
# test_pmids_file = "/home/vs428/project/MedMentions/full/data/corpus_pubtator_pmids_test.txt"

# entity_vocab_file = "/home/vs428/project/MedMentions/full/pretraining5/entity_vocab.jsonl"
# entity_embedding_model = "/home/vs428/project/MedMentions/full/pretraining5/model_epoch20.bin"
# entity_embedding_metadata = "/home/vs428/project/MedMentions/full/pretraining5/metadata.json"

input_data_file = args.input_data_file
train_pmids_file = args.train_pmids
test_pmids_file = args.test_pmids

entity_vocab_file = args.entity_vocab
entity_embedding_model = args.luke_model
entity_embedding_metadata = args.luke_metadata


MAX_LENGTH = args.max_length
MODEL_NAME = args.bert_model_name
num_epochs = args.num_epochs


# Create Tokenizer and DataLoader
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

collater = Collater(tokenizer, max_length = MAX_LENGTH)
train_dataset = MedMentionsDataset(input_data_file, train_pmids_file, tokenizer, entity_vocab_file, 
                             max_length = MAX_LENGTH, stride=0, first_token_ent_only=True)
test_dataset = MedMentionsDataset(input_data_file, test_pmids_file, tokenizer, entity_vocab_file, 
                             max_length = MAX_LENGTH, stride=0, first_token_ent_only=True)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater)
# tokens, bio_tags, entity_ids = next(iter(train_dataloader))



# Retrieve pretrained entity embeddings
model_archive = ModelArchive.load(entity_embedding_model)

model_archive.tokenizer
model_archive.entity_vocab
model_archive.bert_model_name
model_archive.config
model_archive.max_mention_length
pretrained_entity_embeddings = model_archive.state_dict['entity_embeddings.entity_embeddings.weight']
pretrained_entity_embeddings = pretrained_entity_embeddings.to(device)


# define model, etc. 
model = EntityLinker(model_name=MODEL_NAME,
                     pretrained_entity_embeddings=pretrained_entity_embeddings )

model.to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)    
total_training_steps = num_epochs * len(train_dataloader)

if args.lr_scheduler == 'linear':
    optimizer = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_training_steps)
elif args.lr_scheduler == 'cosine':
    optimizer = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_training_steps)
elif args.lr_scheduler == 'warmup':
    optimizer = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps)
elif args.lr_scheduler == 'constant':
    optimizer = get_constant_schedule(optimizer)



# try:
for epoch in range(num_epochs):  # loop over the dataset multiple times
    epoch_start_time = time.time()
    train(model, train_dataloader, optimizer, mention_entity_loss, epoch, pretrained_entity_embeddings, device)
    val_loss = evaluate(model, test_dataloader, optimizer, mention_entity_loss, epoch, pretrained_entity_embeddings, device)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, 
                                      (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)        
# except Exception as e:
#     print("An exception occurred", e) 
    
