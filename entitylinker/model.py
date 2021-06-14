from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F 

from transformers import AutoTokenizer, AutoModel

        
class MentionDetectionHead(nn.Module):
    def __init__(self, input_dim:int):
        '''
        The input_dim must be the size of the embedding dimensions from BERT
        '''
        super(MentionDetectionHead, self).__init__()

        self.linear = nn.Linear(input_dim, 3)

    def forward(self, input: torch.Tensor):
        # input shape: [batch_size, embedding_dim]
        output = self.linear(input)
        return F.log_softmax(output, dim=1)



class EntityDisambiguationHead(nn.Module):
    def __init__(self, input_dim:int, entity_embedding: torch.Tensor):
        '''
        input_dim must be the size of the embedding dimensions from BERT
        entity_embedding is the entity embeddings of size [entity_universe_size, embed_dim]
        '''
        super(EntityDisambiguationHead, self).__init__()

        self.entity_embedding = entity_embedding
        self.entity_embed_dim = self.entity_embedding.shape[1]
        
        self.linear = nn.Linear(input_dim, self.entity_embed_dim)

    def forward(self, input: torch.Tensor):
        '''
        input of shape [batch_size, seq_len, embed_size]
        '''
        output = self.linear(input)
        output = torch.tanh(output)

        # in1 shape [batch_size, seq_len, 1, embed_size]
        # in2 shape [1, entity_vocab, embed_size]
        similarity = F.cosine_similarity(output.unsqueeze(2), self.entity_embedding.unsqueeze(0), dim=-1)
        print("similarity done!")
        return similarity

        

class EntityLinker(nn.Module):
    def __init__(self, model_name: str, pretrained_entity_embeddings:torch.Tensor) -> None:
        super(EntityLinker, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.input_dim = self.pretrained_model.config.hidden_size
        self.md = MentionDetectionHead(self.input_dim)
        self.ed = EntityDisambiguationHead(self.input_dim, pretrained_entity_embeddings)
        

    def forward(self, word_ids: torch.LongTensor, 
                token_type_ids : torch.LongTensor,
                attention_mask : torch.LongTensor):
        print(word_ids)
        print(token_type_ids)
        print(attention_mask)
        # inputs are of form [input_ids, token_type_ids, attention_mask]
        outputs = self.pretrained_model(word_ids, token_type_ids, attention_mask)
        hidden_states = outputs.last_hidden_state

        md_output = self.md(hidden_states)
        ed_output = self.ed(hidden_states)

        return md_output, ed_output






