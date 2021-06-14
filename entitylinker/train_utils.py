import json
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F 
from torch import autograd

import numpy as np 

from transformers.models.luke.configuration_luke import LukeConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig

from entitylinker.lukemodel import LukeModel 
from entitylinker.model_utils import *

from entitylinker.dataset import MedMentionsDataset, Collater
from entitylinker.model import EntityLinker




def sim_matrix(a, b, eps=1e-8):
    """
    works on a 3D and 2D array
    """
    a_n, b_n = torch.linalg.norm(a, dim=-1)[:, :, None], torch.linalg.norm(b, dim=-1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))        
    return sim_mt        


# define loss function
def mention_entity_loss(mention_pred, entity_pred, bio_tags, entity_ids, 
                        attention_mask, pretrained_entity_embedding, device, lm=0.1):
    # attention_mask torch.Size([27,256])
    # mention_pred torch.Size([27, 256, 3])
    # entity_pred torch.Size([27, 256, 256])
    # bio_tags torch.Size([27, 256])
    # entity_ids torch.Size([27, 256])
    
    # unpadded_men_pred torch.Size([25, 3, 256, 256])
    # unpadded_bio_tags torch.Size([25, 256, 256])    
    
    ### MENTION LOSS

    # first compute all loss
    mention_loss = F.nll_loss(mention_pred.permute(0,2,1).contiguous(), bio_tags, reduction="none")

    # get only the unpadded losses: batch size
    num_unpadded = torch.sum(attention_mask, dim=1)
    masked_mention_loss = torch.where(attention_mask == 1, mention_loss, torch.tensor(0.).float().to(device))
    
    # compute average of the masked loss
    avg_mention_loss = torch.sum(masked_mention_loss, dim=1) / num_unpadded
    avg_mention_loss = torch.sum(avg_mention_loss)/len(avg_mention_loss)

    
    ### ENTITY LOSS    
    # get all the IDs in entity_ids that aren't -1
    nonzero_ent_ids = entity_ids[entity_ids!=-1]
    # get the true entity embeddings
    # shape: [nonzero_ent_ids, embed_dim]
    true_ent_embeddings = pretrained_entity_embedding[nonzero_ent_ids, :]
    
    # get non-neg1 idxs 
    ent_idxs = (entity_ids != -1).nonzero()
    real_entity_pred = entity_pred[ent_idxs[:, 0], ent_idxs[:, 1], :]    
    entity_loss = torch.sum(1 - F.cosine_similarity(true_ent_embeddings, real_entity_pred, dim=1))/true_ent_embeddings.shape[0]    
    

    loss = (lm * avg_mention_loss) + ((1-lm) * entity_loss)
    return loss


def get_bi_spans(batch_tags):
    """Returns a list of N x 2 numpy arrays"""
    batch_tags = batch_tags.cpu().detach().numpy()
    bi_spans = []
    for batch_idx in range(batch_tags.shape[0]):
        b_tags = (batch_tags[batch_idx, :] == 0).astype(int)
        i_tags = (batch_tags[batch_idx, :] == 1).astype(int)
        
        # if there aren't any beginning tags, just return an empty array
        if not b_tags.any():
            bi_spans.append(np.array([]))
            continue
        
        # get all B tag locations
        b_idx, = b_tags.nonzero() 

        # get idxs for I tags
        d = np.diff(i_tags)
        i_idx, = d.nonzero()

        # We need to start things after the change in "condition". Therefore, 
        # we'll shift the index by 1 to the right.
        i_idx += 1

        # add end idxs for all B tags (just interweave +1 of b_idxs)
        b_idx = np.vstack((b_idx,b_idx+1)).reshape((-1,),order='F')    

        if i_tags[0]:
            # If the start of condition is True prepend a 0
            i_idx = np.r_[0, i_idx]

        if i_tags[-1]:
            # If the end of condition is True, append the length of the array
            i_idx = np.r_[i_idx, i_tags.size] # Edit

        # reshape to idxs
        b_idx.shape = (-1,2)
        i_idx.shape = (-1,2)    

        bi_idx = []
        # combine the b and i tags
        for start, stop in b_idx:
            # get idx where I tags start for each B tag, if exists
            i, = np.where(i_idx[:, 0] == stop) 
            if i.size > 0:
                bi_idx.append([start, int(i_idx[i, 1])])
            else:
                bi_idx.append([start, stop])

        bi_spans.append(np.array(bi_idx))

    assert batch_tags.shape[0] == len(bi_spans), f"{batch_tags.shape[0]},{len(bi_spans)},{b_idx}"
    return bi_spans


def compute_conf_matrices(mention_preds, entity_preds, bio_tags, entity_ids, pretrained_entity_embedding):
    '''Computes both macro and micro confusion matrices given torch tensor outputs from our model
    '''
    # mention_pred Size([27, 256, 3])
    # entity_pred torch.Size([27, 256, 256])

    nb_classes = 2
    micro_confusion_matrix = torch.zeros(nb_classes, nb_classes)
    # micro_confusion_matrix [[TP, FN], [FP, TN]]
        
    macro_confusion_matrix = []        
    
    # first we need to find the max of the BIO tagging we have
    pred_bio_tags = torch.argmax(mention_preds, dim=2)
    # pred_bio_tags torch.Size([28, 256])
    
    # create two lists of spans
    pred_span_list = get_bi_spans(pred_bio_tags)
    target_span_list = get_bi_spans(bio_tags)    

    for batch_idx, (pred_spans, target_spans) in enumerate(zip(pred_span_list, target_span_list)):
        tp_matches = 0
        
        batch_confusion_matrix = torch.zeros(nb_classes, nb_classes)

        # do the STRONG MATCHING here

        # first check if there are any predicted spans, if not, we just add a bunch of FNs
        if pred_spans.size <= 0 or pred_spans.shape[0] == 0:
            micro_confusion_matrix[0,1] += target_spans.shape[0]
            batch_confusion_matrix[0,1] += target_spans.shape[0]
            
        # conversely, if the target spans are empty, we add a bunch of FPs instead
        elif target_spans.size <= 0:
            micro_confusion_matrix[1,0] += pred_spans.shape[0]
            batch_confusion_matrix[1,0] += pred_spans.shape[0]            
            
        else:
            for span_idx in range(pred_spans.shape[0]):
                
                if torch.all(torch.tensor(pred_spans[span_idx, :] == target_spans), dim=1).any():
                    seq_idx = pred_spans[span_idx, 0]
                    # we use the start span idx to get the entity embedding and compare it to the pretrained embeds

                    cos_sim = sim_matrix(entity_preds[batch_idx, seq_idx, :].unsqueeze(0).unsqueeze(0), 
                                        pretrained_entity_embedding)
                    # this should return something of size [1, 1, sim_vals]
                    pred_ent_embed_id = torch.argmax(torch.squeeze(cos_sim))
                    target_ent_embed_id = entity_ids[batch_idx, seq_idx]
                    # just to test: 
                    # TODO: REMOVE
    #                 if batch_idx == 1 and seq_idx == 0:
    #                     target_ent_embed_id = 2
                    # TODO: ENDREMOVE
                    if pred_ent_embed_id == target_ent_embed_id:
                        # TP
                        micro_confusion_matrix[0,0] += 1
                        batch_confusion_matrix[0,0] += 1 
                        tp_matches += 1
                    else:
                        # FP
                        micro_confusion_matrix[1,0] += 1
                        batch_confusion_matrix[1,0] += 1
                else:
                    # FP 
                    micro_confusion_matrix[1,0] += 1
                    batch_confusion_matrix[1,0] += 1
        
        # all FNs 
        micro_confusion_matrix[0,1] += target_spans.shape[0] - tp_matches
        batch_confusion_matrix[0,1] += target_spans.shape[0] - tp_matches
        
        # we don't do TNs since we don't need them for P,R,F1
        
        # add on the batch confusion matrix to the batch
        macro_confusion_matrix.append(batch_confusion_matrix)

    return micro_confusion_matrix, macro_confusion_matrix


def compute_metrics(metrics, micro_conf, macro_confs):
    '''Given both the micro and macro confusion matrices, we compute a given set of metrics
    
    Matrix orientation is: 
    |  TP  |  FN |
    |-------------
    |  FP  |  TN |
    '''
    micro_metrics = {}
    macro_metrics = {}
    
    if "precision" in metrics:
        micro_metrics['precision'] = micro_conf[0,0] / (micro_conf[0,0] + micro_conf[1,0])
        macro_metrics['precision'] = torch.mean(torch.tensor([batch_conf[0,0] / (batch_conf[0,0] + batch_conf[1,0]) for batch_conf in macro_confs]))
    if "recall" in metrics:
        micro_metrics['recall'] = micro_conf[0,0] / (micro_conf[0,0] + micro_conf[0,1])
        macro_metrics['recall'] = torch.mean(torch.tensor([batch_conf[0,0] / (batch_conf[0,0] + batch_conf[0,1]) for batch_conf in macro_confs]))
    if "f1" in metrics:
        micro_metrics['f1'] = micro_conf[0,0] / (micro_conf[0,0] + ((1/2) * (micro_conf[0,1] + micro_conf[1,0])))
        macro_metrics['f1'] = torch.mean(torch.tensor([batch_conf[0,0] / (batch_conf[0,0] + ((1/2) * (batch_conf[0,1] + batch_conf[1,0]))) for batch_conf in macro_confs]))
        
    return micro_metrics, macro_metrics


# Train!!
def train(model, train_loader, optimizer, criterion, epoch, pretrained_entity_embeddings, device):

    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        tokenized_text, bio_tags, entity_ids = data
        bio_tags = bio_tags.to(device)
        entity_ids = entity_ids.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # with autograd.detect_anomaly():

        # forward + backward + optimize
        input_ids = tokenized_text['input_ids'].to(device)
        token_type_ids = tokenized_text['token_type_ids'].to(device)
        attention_mask = tokenized_text['attention_mask'].to(device)
                          
        mention_pred, entity_pred = model(input_ids,token_type_ids,attention_mask)
        loss = criterion(mention_pred, entity_pred, bio_tags, entity_ids, attention_mask, 
                         pretrained_entity_embeddings, device)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] train loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0
    print('Finished Training')

def evaluate(model, test_loader, optimizer, criterion, epoch, pretrained_entity_embeddings, device):
    
    model.eval()
    total_loss = 0.0
    nb_classes = 2    
    total_micro_conf_mat = torch.zeros(nb_classes, nb_classes)
    # micro_confusion_matrix [[TP, FN], [FP, TN]]
    total_macro_conf_mat = []

    with torch.no_grad():
        running_loss = 0.0
        
        for i, data in enumerate(test_loader, 0):
            
            tokenized_text, bio_tags, entity_ids = data
            bio_tags = bio_tags.to(device)
            entity_ids = entity_ids.to(device)
            
    #         with autograd.detect_anomaly():

            # forward + backward + optimize
            input_ids = tokenized_text['input_ids'].to(device)
            token_type_ids = tokenized_text['token_type_ids'].to(device)
            attention_mask = tokenized_text['attention_mask'].to(device)

            mention_pred, entity_pred = model(input_ids,token_type_ids,attention_mask)
    #         print("ent pred", entity_pred.shape)
            loss = criterion(mention_pred, entity_pred, bio_tags, entity_ids, 
                            attention_mask, pretrained_entity_embeddings, device)
    #         print(loss)


            micro_conf_mat, macro_conf_mat = compute_conf_matrices(mention_pred, entity_pred, bio_tags, entity_ids, pretrained_entity_embeddings)
            total_micro_conf_mat += micro_conf_mat
            total_macro_conf_mat.extend(macro_conf_mat)

            running_loss += loss.item()
            total_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] test loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0


        micro_metrics, macro_metrics = compute_metrics(["precision", "recall","f1"], total_micro_conf_mat, total_macro_conf_mat)
        print('-' * 89)
        print(f"Micro-Precision: {micro_metrics['precision']}, Macro-Precision: {macro_metrics['precision']}\n")
        print(f"Micro-Recall: {micro_metrics['recall']}, Macro-Recall: {macro_metrics['recall']}\n")
        print(f"Micro-F1: {micro_metrics['f1']}, Macro-F1: {macro_metrics['f1']}\n")    
        print('-' * 89)

        # whatever i is at the end is the len of test_loader
        return total_loss / i
