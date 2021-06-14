from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F 
import scipy as sp 

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
        
class MentionDetectionHead(nn.Module):
    def __init__(self, input_dim:int):
        '''
        The input_dim must be the size of the embedding dimensions from BERT
        '''
        super(MentionDetectionHead, self).__init__()

        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 3)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.10)        
        
    def forward(self, input: torch.Tensor):
        # input shape: [batch_size, seq_len]
        output = F.relu(self.linear1(input))
        output = self.dropout(output)
        output = F.relu(self.linear2(output))
        output = self.dropout(output)
        output = F.relu(self.linear4(output))        
        # output shape: [batch_size, seq_len, 3]
        return F.log_softmax(output, dim=2)




class EntityDisambiguationHead(nn.Module):
    def __init__(self, input_dim:int, entity_embedding: torch.Tensor):
        '''
        input_dim must be the size of the embedding dimensions from BERT
        entity_embedding is the entity embeddings of size [entity_universe_size, embed_dim]
        '''
        super(EntityDisambiguationHead, self).__init__()

        self.entity_embedding = entity_embedding
        self.entity_embed_dim = self.entity_embedding.shape[1]
        
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 256)
#         self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, self.entity_embed_dim)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.10)        
        
        
    def forward(self, input_vals: torch.Tensor):
        '''
        input of shape [batch_size, seq_len, embed_size]
        '''
        output = F.relu(self.linear1(input_vals))
        output = self.dropout(output)
        output = F.relu(self.linear2(output))
        output = self.dropout(output)
#         output = F.relu(self.linear3(output))
#         output = self.dropout(output)
        
        output = self.linear4(output)
        output = torch.tanh(output)
        
        # in1 shape [batch_size, seq_len, 1, embed_size]
        # in2 shape [1, entity_vocab, embed_size]
        # output torch.Size([27, 256, 256])        
        # cos_sim torch.Size([27, 256, 34727])
        # batch_size, seq_len = input.shape[0], input.shape[1]
        # ent_vocab_size = self.entity_embedding.shape[0]
        
#         print("output", output.shape)
        return output
#         cos_sim = self._sim_matrix(output, self.entity_embedding)
#         print("cos_sim", cos_sim.shape)
#         max_ent_idxs = torch.argmax(cos_sim, dim=2)
#         max_ents = self.entity_embedding[max_ent_idxs, :]
#         return max_ents

        
#     def _sim_matrix(self, a, b, eps=1e-8):
#         """
#         works on a 3D and 2D array
#         """
#         a_n, b_n = torch.linalg.norm(a, dim=-1)[:, :, None], torch.linalg.norm(b, dim=-1)[:, None]
#         a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#         b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#         sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))        
#         return sim_mt        

class EntityLinker(nn.Module):
    def __init__(self, model_name: str, pretrained_entity_embeddings:torch.Tensor) -> None:
        super(EntityLinker, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.input_dim = self.pretrained_model.config.hidden_size
        self.md = MentionDetectionHead(self.input_dim)
#         print("pretrained_entity_embeddings: ", pretrained_entity_embeddings.shape)
        self.ed = EntityDisambiguationHead(self.input_dim, pretrained_entity_embeddings)
        

    def forward(self, word_ids: torch.LongTensor, 
                token_type_ids : torch.LongTensor,
                attention_mask : torch.LongTensor):
#         print(word_ids)
#         print(token_type_ids)
#         print(attention_mask)
        # inputs are of form [input_ids, token_type_ids, attention_mask]
        outputs = self.pretrained_model(word_ids, token_type_ids, attention_mask)
        hidden_states = outputs.last_hidden_state

        md_output = self.md(hidden_states)
        ed_output = self.ed(hidden_states)

        return md_output, ed_output


    




