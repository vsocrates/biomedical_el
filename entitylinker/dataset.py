from typing import Optional
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from transformers import BertTokenizer 
import json 
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class MedMentionsDataset(Dataset):
    def __init__(self, file_path: str, pmids_path:str, tokenizer: BertTokenizer, entity_vocab_file: str = None, max_length: int = 512, stride:int = 0, first_token_ent_only = False):
        self.file_path = file_path
        self.stride = stride 
        self.max_length = max_length
        self.first_token_ent_only = first_token_ent_only
        
        with open(pmids_path) as f:
            self.pmids = [pmid for pmid in f.read().splitlines()]

        self.entity_vocab = {}
        if entity_vocab_file:
            with open(entity_vocab_file) as f:
                jsonl_list = [json.loads(jline) for jline in f.read().splitlines()]
                for entity in jsonl_list:
                    self.entity_vocab[entity['entities'][0][0]] = entity


        self.data = self._read_data(self.file_path)
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.pmids)

    def __getitem__(self, idx):
        
        pmid = self.pmids[idx]

        data_entry = self.data[pmid]
        text = data_entry['title'] + " " + data_entry['abstract']
        # has [[input_ids], [token_type_ids], [attention_mask], [offset_mapping], overflow_to_sample_mapping]
        # list of lists for overflowing tokens
        tokenized_text = self.tokenizer(text, return_offsets_mapping=True, return_overflowing_tokens=True, return_tensors="pt",
                                        truncation=True, padding = "max_length", max_length=self.max_length, stride=self.stride)
        # print(tokenized_text)
        # we have multiple tokenized text lists
        bio_tags = []
        entity_ids = []
        curr_entity_idx = 0

        for offset_mapping in tokenized_text['offset_mapping']: 

            # start with an Outside (O) tag for the [CLS] token
            chunk_bio_tags = [2]
            beginning_flag = False
            # the [CLS] token is not an entity
            chunk_entity_ids = [-1]

            in_span = False
            # 0: B, 1: I, 2: O
            # skip the [CLS] token
            for token_span_start, token_span_end in offset_mapping[1:].numpy():
                entity = data_entry['entities'][curr_entity_idx]
                entity_start = entity[0]
                entity_end = entity[1]
                # print(token_span_start)
                # print(entity_start)
                # print(token_span_end)
                # print(entity_end)

                # this covers the padded tokens at the end, if any
                if (token_span_start, token_span_end) == (0,0):
                    if self.entity_vocab:
                        chunk_entity_ids.append(-1)

                    chunk_bio_tags.append(2)
                    

                # first we check if the token is in the span and we haven't used a B tag already
                elif (token_span_start >= entity_start) and (token_span_end <= entity_end) and (beginning_flag == False):
                    if self.entity_vocab:
                        chunk_entity_ids.append(self.entity_vocab[entity[4]]['id'])

                    chunk_bio_tags.append(0)
                    beginning_flag = True
                    in_span = True

                # if we get here, we are already in a span, so we want to use an I tag
                elif (token_span_start >= entity_start) and (token_span_end <= entity_end):
                    if self.entity_vocab and not self.first_token_ent_only:
                        chunk_entity_ids.append(self.entity_vocab[entity[4]]['id'])
                    elif self.entity_vocab:
                        chunk_entity_ids.append(-1)
                        
                    chunk_bio_tags.append(1)
                    in_span = True

                # otherwise we're not in a span, and we must either have a Beginning or Outside tag
                # we haven't changed our entity yet so it could be the next entity
                else:
                    # if we're at the end of the entity list, then we must have an Outside tag
                    if (curr_entity_idx + 1) == len(data_entry['entities']):
                        in_span = False
                        beginning_flag = False
                        if self.entity_vocab:
                            chunk_entity_ids.append(-1)

                        chunk_bio_tags.append(2)
                        
                    
                    else:
                        # we'll check if we match the next entity
                        next_entity = data_entry['entities'][curr_entity_idx + 1]
                        next_entity_start = next_entity[0]
                        next_entity_end = next_entity[1]
                        # if we do, we add a Beginning tag and increment the entity counter
                        if (token_span_start >= next_entity_start) and (token_span_end <= next_entity_end):
                            chunk_bio_tags.append(0)
                            if self.entity_vocab:
                                chunk_entity_ids.append(self.entity_vocab[next_entity[4]]['id'])

                            curr_entity_idx += 1
                            in_span = True
                            beginning_flag = True
                            
                        # otherwise we don't and have an Outside Tag
                        else:
                            in_span = False
                            beginning_flag = False
                            if self.entity_vocab:
                                chunk_entity_ids.append(-1)

                            chunk_bio_tags.append(2)
                

            entity_ids.append(chunk_entity_ids)
            bio_tags.append(chunk_bio_tags)
            
            # assert sum([1 for val in chunk_bio_tags if val != 2]) == np.sum(np.array(chunk_entity_ids) >= 0), f"There was an issue in the BIO tagging: {[1 for val in chunk_bio_tags if val == 0]},{np.unique(chunk_entity_ids)}"
            if not self.first_token_ent_only:
                assert sum([1 for val in chunk_bio_tags if val != 2]) == sum([1 for ent in chunk_entity_ids if ent != -1]), f"There was an issue in the BIO tagging: {[1 for val in chunk_bio_tags if val == 0]},{np.unique(chunk_entity_ids)}"
        
        # print(tokenized_text['attention_mask'].shape)
        # print(torch.LongTensor(bio_tags))
        # print(torch.LongTensor(entity_ids))
        if entity_ids:
            # also return the entity IDs 
            return tokenized_text, torch.LongTensor(bio_tags), torch.LongTensor(entity_ids)

        else:
            return tokenized_text, torch.LongTensor(bio_tags)


    def _read_data(self, input_file_path: str):
        data = {}
        
        with open(input_file_path) as f:
            fdata = f.read()
            example_list = fdata.split("\n\n")
        

        for article in example_list:
            article_data = article.split("\n")
            # the first element is the title
            # second is abstract
            # the rest are mentions
            title = article_data[0].split("|") 
            abstract = article_data[1].split("|")
            entities = []
            for entity in article_data[2:]:
                entity_entry = entity.split("\t")[1:]
                # just in case check we don't have an ''
                if not entity_entry:
                    continue
                entity_entry[0] = int(entity_entry[0])
                entity_entry[1] = int(entity_entry[1])
                entities.append(entity_entry)
            
            pmid = title[0]
            data[pmid] = {
                "title": title[2],
                "abstract": abstract[2],
                "entities": entities
            }

        return data


from transformers.data.data_collator import DataCollatorWithPadding


class Collater(object):
  def __init__(self, tokenizer, max_length):
    self.max_length = max_length
    self.tokenizer = tokenizer
    # self.hf_collater = DataCollatorWithPadding(self.tokenizer, padding="max_length", max_length=self.max_length)
  def __call__(self, batch):
    tokenized_text, bio_tags, entity_ids = zip(*batch)
    # print(tokenized_text[0]['position_ids'].shape)
    # print(tokenized_text[1]['position_ids'].shape)

    

    all_input_ids = []    
    all_token_type_ids = []
    all_attention_mask = []
    # all_offset_mapping = []
    # all_overflow_to_sample_mapping = []

    for tokenized_segment in tokenized_text:
        all_input_ids.append(tokenized_segment['input_ids'])
        # print(tokenized_segment['input_ids'].shape)
        all_token_type_ids.append(tokenized_segment['token_type_ids'])
        all_attention_mask.append(tokenized_segment['attention_mask'])
        # tokenized_segment['offset_mapping']
        # tokenized_segment["overflow_to_sample_mapping"]

    # input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,

    tokenized_text_out = {}
    # all_input_ids = self.hf_collater(tokenized_text)
    # tokenized_text_out['input_ids'] = pad_sequence(all_input_ids, batch_first=True)
    # tokenized_text_out['token_type_ids'] = pad_sequence(all_token_type_ids, batch_first=True)
    # tokenized_text_out['attention_mask'] = pad_sequence(all_attention_mask, batch_first=True)
    tokenized_text_out['input_ids'] = torch.cat(all_input_ids)
    tokenized_text_out['token_type_ids'] = torch.cat(all_token_type_ids)
    tokenized_text_out['attention_mask'] = torch.cat(all_attention_mask)
    
    # print(tokenized_text_out['attention_mask'].shape)

    # bio_tags = pad_sequence(bio_tags, batch_first=True, padding_value=2)
    # entity_ids = pad_sequence(entity_ids, batch_first=True, padding_value=-1)
    bio_tags = torch.cat(bio_tags)
    entity_ids = torch.cat(entity_ids)

    # print(entity_ids.shape)
    
    # now you can apply your sequence padding to the whole batch.
    # tokenized_text = self.hf_collater(tokenized_text)

    # print(bio_tags[0].shape)
    # print(bio_tags[1].shape)
    # print(bio_tags.shape)

    # bio_tags = pad_sequence(bio_tags, batch_first=True)
    # print(bio_tags.shape)
    # entity_ids = pad_sequence(entity_ids, batch_first=True)
    return tokenized_text_out, bio_tags, entity_ids



