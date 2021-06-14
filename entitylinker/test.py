# def sim_matrix(a, b, eps=1e-8):
#     """
#     works on a 3D and 2D array
#     """
#     a_n, b_n = torch.linalg.norm(a, dim=-1)[:, :, None], torch.linalg.norm(b, dim=-1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.matmul(a_norm, b_norm.transpose(0, 1))        
#     return sim_mt        



# def get_bi_spans(batch_tags):
#     """Returns a N x 2 numpy array"""
#     batch_tags = batch_tags.cpu().detach().numpy()
#     bi_spans = []
#     for batch_idx in range(batch_tags.shape[0]):
#         b_tags = (batch_tags[batch_idx, :] == 0).astype(int)
#         i_tags = (batch_tags[batch_idx, :] == 1).astype(int)
        
#         # if there aren't any beginning tags, just return an empty array
#         if not b_tags.any():
#             bi_spans.append(np.array([]))
        
#         # get all B tag locations
#         b_idx, = b_tags.nonzero() 

#         # get idxs for I tags
#         d = np.diff(i_tags)
#         i_idx, = d.nonzero()

#         # We need to start things after the change in "condition". Therefore, 
#         # we'll shift the index by 1 to the right.
#         i_idx += 1

#         # add end idxs for all B tags (just interweave +1 of b_idxs)
#         b_idx = np.vstack((b_idx,b_idx+1)).reshape((-1,),order='F')    

#         if i_tags[0]:
#             # If the start of condition is True prepend a 0
#             i_idx = np.r_[0, i_idx]

#         if i_tags[-1]:
#             # If the end of condition is True, append the length of the array
#             i_idx = np.r_[i_idx, i_tags.size] # Edit

#         # reshape to idxs
#         b_idx.shape = (-1,2)
#         i_idx.shape = (-1,2)    

#         bi_idx = []
#         # combine the b and i tags
#         for start, stop in b_idx:
#             # get idx where I tags start for each B tag, if exists
#             i, = np.where(i_idx[:, 0] == stop) 
#             if i.size > 0:
#                 bi_idx.append([start, int(i_idx[i, 1])])
#             else:
#                 bi_idx.append([start, stop])

#         bi_spans.append(np.array(bi_idx))

#     return bi_spans


# def compute_conf_matrices(mention_preds, entity_preds, bio_tags, entity_ids, pretrained_entity_embedding):
#     '''Computes both macro and micro confusion matrices given torch tensor outputs from our model
#     '''
#     # mention_pred Size([27, 256, 3])
#     # entity_pred torch.Size([27, 256, 256])

#     nb_classes = 2
#     micro_confusion_matrix = torch.zeros(nb_classes, nb_classes)
#     # micro_confusion_matrix [[TP, FN], [FP, TN]]
        
#     macro_confusion_matrix = []        
    
#     # first we need to find the max of the BIO tagging we have
#     pred_bio_tags = torch.argmax(mention_preds, dim=2)
#     # pred_bio_tags torch.Size([28, 256])
    
#     # create two lists of spans
#     pred_span_list = get_bi_spans(pred_bio_tags)
#     target_span_list = get_bi_spans(bio_tags)    
    
#     for batch_idx, (pred_spans, target_spans) in enumerate(zip(pred_span_list, target_span_list)):
#         tp_matches = 0
        
#         batch_confusion_matrix = torch.zeros(nb_classes, nb_classes)

#         # do the STRONG MATCHING here

#         # first check if there are any predicted spans, if not, we just add a bunch of FNs
#         if pred_spans.size <= 0 or pred_spans.shape[0] == 0:
#             micro_confusion_matrix[0,1] += target_spans.shape[0]
#             batch_confusion_matrix[0,1] += target_spans.shape[0]
            
#         # conversely, if the target spans are empty, we add a bunch of FPs instead
#         elif target_spans.size <= 0:
#             micro_confusion_matrix[1,0] += pred_spans.shape[0]
#             batch_confusion_matrix[1,0] += pred_spans.shape[0]            
            
#         else:
#             for span_idx in range(pred_spans.shape[0]):
                
#                 if torch.all(torch.tensor(pred_spans[span_idx, :] == target_spans), dim=1).any():
#                     seq_idx = pred_spans[span_idx, 0]
#                     # we use the start span idx to get the entity embedding and compare it to the pretrained embeds

#                     cos_sim = sim_matrix(entity_preds[batch_idx, seq_idx, :].unsqueeze(0).unsqueeze(0), 
#                                         pretrained_entity_embedding)
#                     # this should return something of size [1, 1, sim_vals]
#                     pred_ent_embed_id = torch.argmax(torch.squeeze(cos_sim))
#                     target_ent_embed_id = entity_ids[batch_idx, seq_idx]
#                     # just to test: 
#                     # TODO: REMOVE
#     #                 if batch_idx == 1 and seq_idx == 0:
#     #                     target_ent_embed_id = 2
#                     # TODO: ENDREMOVE
#                     if pred_ent_embed_id == target_ent_embed_id:
#                         # TP
#                         micro_confusion_matrix[0,0] += 1
#                         batch_confusion_matrix[0,0] += 1 
#                         tp_matches += 1
#                     else:
#                         # FP
#                         micro_confusion_matrix[1,0] += 1
#                         batch_confusion_matrix[1,0] += 1
#                 else:
#                     # FP 
#                     micro_confusion_matrix[1,0] += 1
#                     batch_confusion_matrix[1,0] += 1
        
#         # all FNs 
#         micro_confusion_matrix[0,1] += target_spans.shape[0] - tp_matches
#         batch_confusion_matrix[0,1] += target_spans.shape[0] - tp_matches
        
#         # we don't do TNs since we don't need them for P,R,F1
        
#         # add on the batch confusion matrix to the batch
#         macro_confusion_matrix.append(batch_confusion_matrix)

#     return micro_confusion_matrix, macro_confusion_matrix




# def compute_metrics(metrics, micro_conf, macro_confs):
#     '''Given both the micro and macro confusion matrices, we compute a given set of metrics
    
#     Matrix orientation is: 
#     |  TP  |  FN |
#     |-------------
#     |  FP  |  TN |
#     '''
#     micro_metrics = {}
#     macro_metrics = {}
    
#     if "precision" in metrics:
#         micro_metrics['precision'] = micro_conf[0,0] / (micro_conf[0,0] + micro_conf[1,0])
#         macro_metrics['precision'] = torch.tensor([batch_conf[0,0] / (batch_conf[0,0] + batch_conf[1,0]) for batch_conf in macro_confs])/len(macro_confs)
#     if "recall" in metrics:
#         micro_metrics['recall'] = micro_conf[0,0] / (micro_conf[0,0] + micro_conf[0,1])
#         macro_metrics['recall'] = torch.tensor([batch_conf[0,0] / (batch_conf[0,0] + batch_conf[0,1]) for batch_conf in macro_confs])/len(macro_confs)
#     if "f1" in metrics:
#         micro_metrics['f1'] = micro_conf[0,0] / (micro_conf[0,0] + ((1/2) * (micro_conf[0,1] + micro_conf[1,0])))
#         macro_metrics['f1'] = torch.tensor([micro_conf[0,0] / (batch_conf[0,0] + ((1/2) * (batch_conf[0,1] + batch_conf[1,0]))) for batch_conf in macro_confs]) / len(macro_confs)
        
#     return micro_metrics, macro_metrics

########################################################################################################
########################################################################################################
########################################################################################################

# define loss function
def mention_entity_loss(mention_pred, entity_pred, bio_tags, entity_ids, 
                        attention_mask, pretrained_entity_embedding, lm=0.1):
    # attention_mask torch.Size([27,256])
    # mention_pred torch.Size([27, 256, 3])
    # entity_pred torch.Size([27, 256, 256])
    # bio_tags torch.Size([27, 256])
    # entity_ids torch.Size([27, 256])
    
    # unpadded_men_pred torch.Size([25, 3, 256, 256])
    # unpadded_bio_tags torch.Size([25, 256, 256])    
    
    ### MENTION LOSS

    # first compute all loss
    # mention_loss = F.nll_loss(mention_pred.permute(0,2,1).contiguous(), bio_tags, reduction="none")
    print("test", torch.FloatTensor(mention_pred).permute(0,2,1).contiguous().shape)
    test  = F.log_softmax(torch.FloatTensor(mention_pred).permute(0,2,1).contiguous(), dim=1)
    print(torch.LongTensor(bio_tags).shape)
    print(test.exp())
    mention_loss = F.nll_loss(test, 
        torch.LongTensor(bio_tags), reduction="none")
    # get only the unpadded losses: batch size
    num_unpadded = torch.sum(torch.FloatTensor(attention_mask), dim=1)
    masked_mention_loss = torch.where(torch.tensor(attention_mask == 1, dtype=bool), mention_loss, torch.tensor(0.).float())
    
    # compute average of the masked loss
    avg_mention_loss = torch.sum(masked_mention_loss, dim=1) / num_unpadded
    avg_mention_loss = torch.sum(avg_mention_loss)/len(avg_mention_loss)

#     print("MENTION LOSS: ", avg_mention_loss)

    
    ### ENTITY LOSS
#     cos_sim = sim_matrix(entity_pred, pretrained_entity_embedding)
#     print("cos_sim", cos_sim.shape)
    # cos_sim torch.Size([27, 256, 34727])
    
    # get all the IDs in entity_ids that aren't -1
    nonzero_ent_ids = entity_ids[entity_ids!=-1]
    # get the true entity embeddings
    # shape: [nonzero_ent_ids, embed_dim]
    true_ent_embeddings = torch.tensor(pretrained_entity_embedding[nonzero_ent_ids, :])
    
    # get non-neg1 idxs 
    ent_idxs = (torch.LongTensor(entity_ids) != -1).nonzero()
    real_entity_pred = entity_pred[ent_idxs[:, 0], ent_idxs[:, 1], :]    
#     print("true_ent_embeddings", true_ent_embeddings.shape)
    true_ent_embeddings = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    print("SDFSDFS", 1 - F.cosine_similarity(true_ent_embeddings, torch.tensor(real_entity_pred), dim=1))
    entity_loss = torch.sum(1 - F.cosine_similarity(true_ent_embeddings, torch.tensor(real_entity_pred), dim=1))/true_ent_embeddings.shape[0]
#     print("ENTITY LOSS", entity_loss)
#     max_ents = torch.max(cos_sim, dim=2)
    
    # target_ent_embeddings = pretrained_entity_embedding[entity_ids, :]
    # target_ent_embeddings torch.Size([27, 256, 256])
    # max_ent_idxs = torch.argmax(cos_sim, dim=2)
#     max_ents = self.entity_embedding[max_ent_idxs, :]
    
    
#     entity_pred[ent_idxs]
#     entity_loss = torch.mean((output - target)**2)
    
    

    loss = (lm * avg_mention_loss) + ((1-lm) * entity_loss)
    return loss

batch_size = 2
ent_vocab_size = 8
import torch.nn.functional as F
import torch 
import numpy as np

torch.manual_seed(1)
np.random.seed(1)

softmax = torch.nn.LogSoftmax(dim=1)
print(softmax(torch.FloatTensor([[1,0],[2,0]])))
print(torch.FloatTensor([[1,0,0],[1,0,0]]).shape)
mention_loss = F.nll_loss(softmax(torch.FloatTensor([[1,0],[2,0]])), 
        torch.LongTensor([0 , 0])
        )
print(mention_loss)

metric_list = ""
# mention_preds = torch.FloatTensor(np.random.rand(batch_size, 5, 3))
# mention_preds = torch.FloatTensor(np.random.rand(batch_size, 5, 3))
mention_preds = np.array([ [[-100.,0,0], [0,-100,0]], [[-100,0,0], [0,0,-100]] ], dtype=float)
print("mention_preds", mention_preds.shape)
# mention_preds = F.softmax(torch.tensor(mention_preds), dim=2)
# print(mention_preds)
# entity_preds = torch.FloatTensor(np.random.rand(batch_size, 5, 5))
entity_preds = np.array([ [[1.,0,0], [0,0.,0]], [[0.,1,0], [0,0,0.]] ], dtype=float)
print(entity_preds.shape)
# bio_tags = torch.tensor(np.random.randint(0,3, size=(batch_size, 5)))

# bio_tags = torch.tensor([[0, 0, 1, 0, 0],
#         [0, 2, 0, 2, 2]])
bio_tags = np.array([[0, 1],[0, 2]], dtype=float)
# bio_tags = torch.tensor([ [[1,0,0], [0,0,0]], [[0,1,0], [0,0,0]] ])
# entity_ids = torch.tensor([[4, 3, -1, 1, 1],
#         [6, -1, 3, -1, -1]])
entity_ids = np.array([[4, -1],[6, -1]])

# attention_mask = torch.FloatTensor([[1,1,1,1,1],[1,1,1,1,1]])
attention_mask = np.array([[1,0],[1,1]])

######################################################################
print(bio_tags.shape)
print(bio_tags)
# entity_ids = torch.tensor(np.random.randint(ent_vocab_size, size=(batch_size, 5)))
print(entity_ids)
pretrained_entity_embedding = torch.FloatTensor(np.random.rand(ent_vocab_size, 3))
print(pretrained_entity_embedding.shape)
# mention_pred Size([27, 256, 3])
# entity_pred torch.Size([27, 256, 256])
# bio_tags torch.Size([27, 256])
# entity_ids torch.Size([27, 256])

loss =mention_entity_loss(mention_preds, entity_preds, bio_tags, entity_ids, attention_mask, pretrained_entity_embedding)
print(f"loss: {loss}")
# #########################
# micro_conf_mat, macro_conf_mat = compute_conf_matrices(mention_preds, entity_preds, bio_tags, entity_ids, pretrained_entity_embedding)
# print(micro_conf_mat)
# print(macro_conf_mat)
# micro_metrics, macro_metrics = compute_metrics(["precision", "recall","f1"], micro_conf_mat, macro_conf_mat)
# print(micro_metrics)
# print(macro_metrics)
#########################



# data = {'input_ids': [[2, 10, 17915, 1883, 10, 61, 1765, 42, 5122, 1690, 42, 4022, 5179, 59, 10, 29, 1680, 2626, 16, 2978, 6215, 1685, 10539, 1682, 3891, 16, 42, 6651, 1901, 3357, 1680, 2626, 16, 2978, 6215, 1685, 10539, 1682, 3891, 1682, 1680, 8292, 1814, 1680, 11280, 1685, 12803, 1783, 1690, 16851, 17, 6651, 25210, 3493, 2345, 1690, 7210, 1715, 16851, 1690, 12803, 1783, 17, 24426, 3045, 3891, 1682, 18570, 15, 12984, 4393, 2950, 1690, 2226, 2318, 15, 18468, 1016, 15, 1930, 23739, 1690, 2969, 4240, 1041, 15, 1690, 20081, 2098, 1680, 20703, 13325, 11561, 4756, 2372, 1798, 1680, 2188, 1685, 7757, 17, 3100, 1685, 42, 4595, 3135, 3045, 17775, 1901, 2754, 1680, 10526, 2659, 1685, 24426, 3045, 3891, 17, 4950, 16851, 1690, 4464, 12803, 1783, 17, 16851, 8658, 8292, 19855, 1690, 5916, 3045, 19855, 15, 3850, 1948, 4745, 1685, 3891, 1682, 25, 20972, 11, 3135, 3802, 15, 4921, 15, 19921, 1690, 19417, 2317, 15, 15203, 15, 14750, 15, 2868, 12, 1690, 12133, 17, 24426, 3891, 1734, 11851, 1772, 2626, 16, 2978, 13603, 17, 2476, 4086, 1798, 2626, 1748, 7216, 9206, 1715, 10099, 1690, 9745, 1685, 1680, 46, 16, 5908, 12012, 13300, 1025, 17, 16851, 13178, 16174, 14084, 1026, 1950, 2509, 15861, 2076, 6442, 11529, 1690, 9324, 4626, 1701, 6021, 1814, 15212, 1690, 3372, 17, 1805, 3621, 2021, 7795, 1701, 16612, 1715, 6442, 8749, 15, 1680, 11548, 1685, 1950, 2509, 3378, 21490, 1007, 1755, 4679, 1690, 3891, 15, 2555, 2082, 14291, 16851, 1715, 42, 3805, 1685, 3597, 3011, 8264, 2626, 1701, 8574, 3012, 1680, 6394, 11548, 1685, 3891, 17, 2052, 15175, 1007, 1690, 8698, 1701, 2300, 16, 3203, 3745, 1682, 2585, 16851, 16958, 1894, 18306, 4804, 3198, 17, 2626, 16, 2978, 13603, 1734, 4124, 3258, 1725, 2415, 1715, 2564, 1690, 3882, 2592, 2386, 15591, 42, 3805, 1685, 2552, 16, 1981, 16, 4591, 16, 2194, 7561, 1690, 23883, 7814, 17, 3653, 16851, 4534, 13915, 1678, 2754, 4377, 1701, 15522, 42, 14780, 2639, 13146, 1734, 2051, 2986, 2014, 1701, 42, 2626, 16, 2978, 6215, 2330, 1732, 3135, 3802, 17, 1680, 3733, 20703, 13325, 11561, 1734, 13146, 1701, 23962, 5639, 4159, 17, 42, 3805, 1685, 2626, 16, 2978, 6215, 1682, 24426, 3045, 3891, 7004, 14810, 1755, 16851, 10, 4679, 1690, 2300, 16, 3203, 17, 2592, 1715, 2564, 1748, 4124, 3730, 15, 3814, 1805, 1922, 3722, 5090, 1680, 8103, 1814, 2458, 1701, 2814, 1680, 2626, 16, 2978, 6215, 1685, 16851, 17, 3]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'offset_mapping': [[(0, 0), (0, 1), (1, 4), (5, 8), (8, 9), (9, 10), (11, 13), (14, 15), (16, 22), (23, 26), (27, 28), (29, 32), (32, 34), (35, 36), (36, 37), (37, 38), (39, 42), (43, 47), (47, 48), (48, 52), (53, 60), (61, 63), (64, 71), (72, 74), (75, 83), (84, 85), (85, 86), (87, 98), (99, 104), (105, 116), (117, 120), (121, 125), (125, 126), (126, 130), (131, 138), (139, 141), (142, 149), (150, 152), (153, 161), (162, 164), (165, 168), (169, 171), (172, 176), (177, 180), (181, 193), (194, 196), (197, 202), (202, 205), (206, 209), (210, 218), (218, 219), (220, 231), (232, 246), (247, 252), (253, 259), (260, 263), (264, 274), (275, 279), (280, 288), (289, 292), (293, 298), (298, 301), (301, 302), (303, 315), (316, 323), (324, 332), (333, 335), (336, 342), (342, 343), (344, 348), (348, 350), (350, 353), (354, 357), (358, 361), (361, 364), (364, 365), (366, 369), (369, 370), (370, 371), (372, 375), (375, 378), (379, 382), (383, 386), (386, 388), (388, 389), (389, 390), (391, 394), (395, 400), (401, 407), (408, 411), (412, 418), (419, 425), (426, 434), (435, 439), (439, 442), (443, 445), (446, 449), (450, 453), (454, 456), (457, 461), (461, 462), (463, 467), (468, 470), (471, 472), (473, 479), (480, 487), (488, 495), (496, 503), (504, 509), (510, 515), (516, 519), (520, 524), (524, 528), (529, 531), (532, 544), (545, 552), (553, 561), (561, 562), (563, 565), (566, 574), (575, 578), (579, 581), (582, 587), (587, 590), (590, 591), (592, 600), (601, 610), (611, 613), (614, 623), (624, 627), (628, 641), (642, 649), (650, 659), (659, 660), (661, 667), (668, 671), (672, 678), (679, 681), (682, 690), (691, 693), (694, 695), (696, 707), (708, 709), (709, 716), (717, 725), (725, 726), (727, 735), (735, 736), (737, 747), (748, 751), (752, 758), (758, 763), (763, 764), (765, 775), (775, 776), (777, 786), (786, 787), (788, 795), (795, 796), (797, 800), (801, 811), (811, 812), (813, 825), (826, 834), (835, 838), (839, 852), (853, 855), (856, 860), (860, 861), (861, 865), (866, 875), (875, 876), (877, 881), (882, 887), (888, 890), (891, 895), (896, 900), (901, 910), (911, 923), (924, 928), (929, 937), (938, 941), (942, 952), (953, 955), (956, 959), (960, 961), (961, 962), (962, 966), (966, 968), (968, 970), (970, 971), (971, 972), (973, 981), (982, 991), (992, 997), (998, 1007), (1007, 1008), (1009, 1014), (1015, 1020), (1021, 1028), (1029, 1034), (1035, 1043), (1044, 1049), (1050, 1053), (1054, 1063), (1064, 1067), (1068, 1070), (1071, 1081), (1082, 1086), (1087, 1094), (1095, 1098), (1099, 1105), (1105, 1106), (1107, 1111), (1112, 1116), (1117, 1119), (1120, 1131), (1132, 1134), (1135, 1139), (1140, 1144), (1145, 1153), (1154, 1163), (1163, 1164), (1165, 1168), (1169, 1177), (1178, 1180), (1181, 1186), (1187, 1192), (1193, 1197), (1198, 1204), (1204, 1205), (1206, 1208), (1209, 1217), (1218, 1221), (1222, 1230), (1230, 1231), (1232, 1237), (1238, 1242), (1243, 1250), (1251, 1259), (1260, 1264), (1265, 1266), (1267, 1271), (1272, 1274), (1275, 1281), (1282, 1289), (1290, 1297), (1298, 1302), (1303, 1305), (1306, 1312), (1313, 1320), (1321, 1324), (1325, 1337), (1338, 1346), (1347, 1349), (1350, 1358), (1358, 1359), (1360, 1363), (1364, 1369), (1369, 1370), (1371, 1374), (1375, 1379), (1380, 1382), (1383, 1387), (1387, 1388), (1388, 1393), (1394, 1402), (1403, 1405), (1406, 1410), (1411, 1419), (1420, 1427), (1428, 1430), (1430, 1432), (1432, 1435), (1435, 1439), (1439, 1440), (1441, 1445), (1445, 1446), (1446, 1450), (1451, 1460), (1461, 1464), (1465, 1477), (1478, 1484), (1485, 1488), (1489, 1494), (1495, 1499), (1500, 1508), (1509, 1512), (1513, 1523), (1524, 1529), (1530, 1533), (1534, 1539), (1540, 1541), (1542, 1546), (1547, 1549), (1550, 1554), (1554, 1555), (1555, 1559), (1560, 1561), (1562, 1566), (1566, 1567), (1567, 1571), (1572, 1581), (1582, 1585), (1586, 1600), (1601, 1610), (1610, 1611), (1612, 1618), (1619, 1627), (1628, 1638), (1639, 1643), (1643, 1645), (1646, 1651), (1652, 1658), (1659, 1661), (1662, 1668), (1669, 1670), (1671, 1680), (1681, 1685), (1686, 1690), (1691, 1694), (1695, 1699), (1700, 1706), (1706, 1709), (1710, 1712), (1713, 1714), (1715, 1719), (1719, 1720), (1720, 1724), (1725, 1732), (1733, 1737), (1738, 1740), (1741, 1748), (1749, 1757), (1757, 1758), (1759, 1762), (1763, 1771), (1772, 1778), (1779, 1785), (1786, 1794), (1795, 1798), (1799, 1803), (1804, 1806), (1807, 1817), (1818, 1826), (1827, 1835), (1835, 1836), (1837, 1838), (1839, 1843), (1844, 1846), (1847, 1851), (1851, 1852), (1852, 1856), (1857, 1864), (1865, 1867), (1868, 1880), (1881, 1888), (1889, 1897), (1898, 1908), (1909, 1917), (1918, 1920), (1921, 1929), (1930, 1931), (1932, 1940), (1941, 1944), (1945, 1949), (1949, 1950), (1950, 1955), (1955, 1956), (1957, 1962), (1963, 1967), (1968, 1976), (1977, 1981), (1982, 1994), (1995, 2003), (2003, 2004), (2005, 2015), (2016, 2020), (2021, 2026), (2027, 2032), (2033, 2040), (2041, 2044), (2045, 2053), (2054, 2058), (2059, 2066), (2067, 2069), (2070, 2077), (2078, 2081), (2082, 2086), (2086, 2087), (2087, 2091), (2092, 2099), (2100, 2102), (2103, 2111), (2111, 2112), (0, 0)]]}
# for input_ids, token_type_ids, attention_mask, offset_mapping in zip(data['input_ids'],data['token_type_ids'] , data['attention_mask'], data['offset_mapping']):
#     print(input_ids)
    

# import torch
# data = torch.LongTensor([[[   0,    0],
#          [   0,   11],
#          [  12,   14],
#          [  15,   18],
#          [  18,   20],
#          [  20,   21],
#          [  21,   22],
#          [  23,   27],
#          [  27,   29],
#          [   0,    0]],
#         [[   0,    0],
#          [  30,   31],
#          [  32,   33],
#          [  34,   35],
#          [  36,   38],
#          [  38,   40],
#          [  41,   45],
#          [  46,   59],
#          [  60,   64],
#          [   0,    0]]]   ) 
# for x in data:
#     for a,b in x.numpy():
#         print(a)
#         print("new \n")

# import sys
# sys.exit()
####################################################################

# import json
# import torch 
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from transformers.models.luke.configuration_luke import LukeConfig
# from transformers import AutoTokenizer, AutoModel, AutoConfig

# from lukemodel import LukeModel 
# from model_utils import *

# from dataset import MedMentionsDataset, Collater
# from model import EntityLinker 

# input_data_file = "/Users/vsocrates/Documents/Yale/EntityLinking/data/MedMentions/full/data/corpus_pubtator.txt"
# pmids_file = "/Users/vsocrates/Documents/Yale/EntityLinking/data/MedMentions/full/data/corpus_pubtator_pmids_all.txt"
# entity_vocab_file = "/Users/vsocrates/Documents/Yale/EntityLinking/data/MedMentions/full/entity_vocab.jsonl"




# MAX_LENGTH = 256
# MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# collater = Collater(tokenizer, max_length = MAX_LENGTH)
# dataset = MedMentionsDataset(input_data_file, pmids_file, tokenizer, entity_vocab_file, max_length = MAX_LENGTH, stride=0)
# train_dataloader = DataLoader(dataset, batch_size=15, shuffle=True, collate_fn=collater)
# tokens, bio_tags, entity_ids = next(iter(train_dataloader))

# print(tokens['input_ids'].shape)
# print(bio_tags.shape)
# print(entity_ids.shape)
# print(bio_tags.shape)
# print(entity_ids.shape)

# # define optimizer, model, loss etc. 

# entity_embedding_model = "/Users/vsocrates/Documents/Yale/EntityLinking/data/MedMentions/full/model_epoch20.bin"
# entity_embedding_metadata = "/Users/vsocrates/Documents/Yale/EntityLinking/data/MedMentions/full/metadata.json"

# # with open(entity_embedding_metadata) as f:
# #     luke_metadata = json.loads(f.read())

# # luke_config = luke_metadata['model_config']
# # bert_config = AutoConfig.from_pretrained(MODEL_NAME)

# # config = LukeConfig(
# #     entity_vocab_size=luke_config['entity_vocab_size'],
# #     bert_model_name=luke_config['bert_model_name'],
# #     entity_emb_size=luke_config['entity_emb_size'],
# #     **bert_config.to_dict(),
# # )

# # luke_model = LukeModel(config)
# # pretrained_entity_embeddings = luke_model.load_state_dict(torch.load(entity_embedding_model, map_location=torch.device('cpu')))
# # print(pretrained_entity_embeddings)

# model_archive = ModelArchive.load(entity_embedding_model)

# model_archive.tokenizer
# model_archive.entity_vocab
# model_archive.bert_model_name
# model_archive.config
# model_archive.max_mention_length
# pretrained_entity_embeddings = model_archive.state_dict['entity_embeddings.entity_embeddings.weight']

# model = EntityLinker(model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
#                      pretrained_entity_embeddings=pretrained_entity_embeddings )
# def my_loss(output, target):
#     loss = torch.mean((output - target)**2)
#     return loss

# criterion = my_loss
# optimizer = optim.Adam(model.parameters(), lr=0.00002)


# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(train_dataloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         tokenized_text, bio_tags, entity_ids = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(tokenized_text['input_ids'],
#                         tokenized_text['token_type_ids'],
#                         tokenized_text['attention_mask'], )
#         print(outputs[0])
#         print(outputs[1])
#         # loss = criterion(outputs, labels)
#         # loss.backward()
#         # optimizer.step()

#         # # print statistics
#         # running_loss += loss.item()
#         # if i % 2000 == 1999:    # print every 2000 mini-batches
#         #     print('[%d, %5d] loss: %.3f' %
#         #           (epoch + 1, i + 1, running_loss / 2000))
#         #     running_loss = 0.0

# print('Finished Training')


####################################################################
# import transformers
# from transformers import AutoTokenizer

# input_data_file = "/Users/vsocrates/Documents/Yale/EntityLinking/data/MedMentions/full/data/corpus_pubtator.txt"

# data = {}
# with open(input_data_file) as f:
#     fdata = f.read()
#     example_list = fdata.split("\n\n")


# for article in example_list:
#     article_data = article.split("\n")
#     # the first element is the title
#     # second is abstract
#     # the rest are mentions
#     title = article_data[0].split("|") 
#     # print(article_data[1])
#     abstract = article_data[1].split("|")
#     entities = []
#     for entity in article_data[2:]:
#         entity_entry = entity.split("\t")[1:]
#         # just in case check we don't have an ''
#         if not entity_entry:
#             continue
#         entity_entry[0] = int(entity_entry[0])
#         entity_entry[1] = int(entity_entry[1])
#         entities.append(entity_entry)
    
#     pmid = title[0]
#     data[pmid] = {
#         "title": title[2],
#         "abstract": abstract[2],
#         "entities": entities
#     }

# token_len = []
# import numpy as np
# tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')

# for key, value in data.items():
#     fulltext = value['title'] + " " + value['abstract']
#     vals = tokenizer(fulltext)
#     # print(vals)
#     token_len.append(len(vals['input_ids']))
#     np_token_len = np.array(token_len)
# print(np.sum(np_token_len >= 512) / np_token_len.shape[0])
# # print(token_len)
# # print(pmid)


#######
# import json 

# with open("/Users/vsocrates/Documents/Yale/EntityLinking/luke/tests/dataset/entity_vocab.jsonl") as f:
#     output = [json.loads(jline) for jline in f.readlines()]
# print(output[0])
# import sys
# sys.exit()
#######

####################################################################
# import transformers
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
# output = tokenizer(data["28104446"]['title'] + " " + data["28104446"]['abstract'], return_offsets_mapping=True)
# # print(output)
# curr_entity_idx = 0
# # start with an Outside (O) tag for the [CLS] token
# output_bio_tags = [2]
# beginning_flag = False
# in_span = False
# # skip the [CLS] token and [SEP] token
# # 0: B, 1: I, 2: O
# for token_span_start, token_span_end in output.offset_mapping[1:-1]:
#     entity = data["28104446"]['entities'][curr_entity_idx]
#     entity_start = entity[0]
#     entity_end = entity[1]
#     # print(token_span_start)
#     # print(entity_start)
#     # print(token_span_end)
#     # print(entity_end)

#     # first we check if the token is in the span and we haven't used a B tag already
#     if (token_span_start >= entity_start) and (token_span_end <= entity_end) and (beginning_flag == False):
#         output_bio_tags.append(0)
#         beginning_flag = True
#         in_span = True
#     # if we get here, we are already in a span, so we want to use an I tag
#     elif (token_span_start >= entity_start) and (token_span_end <= entity_end):
#         output_bio_tags.append(1)
#         in_span = True
#     # otherwise we're not in a span, and we must either have a Beginning or Outside tag
#     # we haven't changed our entity yet so it could be the next entity
#     else:
#         # if we're at the end of the entity list, then we must have an Outside tag
#         if (curr_entity_idx + 1) == len(data["28104446"]['entities']):
#             in_span = False
#             beginning_flag = False
#             output_bio_tags.append(2)
#             continue
        
#         else:
#             # we'll check if we match the next entity
#             next_entity = data["28104446"]['entities'][curr_entity_idx + 1]
#             next_entity_start = next_entity[0]
#             next_entity_end = next_entity[1]
#             # if we do, we add a Beginning tag and increment the entity counter
#             if (token_span_start >= next_entity_start) and (token_span_end <= next_entity_end):
#                 output_bio_tags.append(0)
#                 curr_entity_idx += 1
#                 in_span = True
#                 beginning_flag = True
#                 continue
#             # otherwise we don't and have an Outside Tag
#             else:
#                 in_span = False
#                 beginning_flag = False
#                 output_bio_tags.append(2)

# # end with an Outside (O) tag for the [SEP] token
# output_bio_tags.append(2)
# print(output_bio_tags)
# print(sum([1 for val in output_bio_tags if val == 0]))
# print(len(data["28104446"]['entities']))

    # if curr_entity_idx == 7:
    #     break


####################################################################
# import transformers
# from transformers import AutoTokenizer
# import numpy as np

# test_pmid = "27622108"


# tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
# output = tokenizer(data[test_pmid]['title'] + " " + data[test_pmid]['abstract'], return_offsets_mapping=True, return_overflowing_tokens=True,# return_tensors="pt",
#                                         truncation=True, padding = "max_length", max_length=10, stride=0)

# bio_tags = []
# entity_ids = []
# prev_final_entity_idx = 0
# curr_entity_idx = 0
# entity_vocab = True
# for input_ids, offset_mapping in zip(output['input_ids'], output['offset_mapping']): 
    
#     # start with an Outside (O) tag for the [CLS] token
#     chunk_bio_tags = [2]
#     beginning_flag = False
#     # the [CLS] token is not an entity
#     chunk_entity_ids = [-1]

#     in_span = False
#     # 0: B, 1: I, 2: O
#     # skip the [CLS] token
#     for token_span_start, token_span_end in offset_mapping[1:]:
#         entity = data[test_pmid]['entities'][curr_entity_idx]
#         entity_start = entity[0]
#         entity_end = entity[1]
#         # print(token_span_start)
#         # print(entity_start)
#         # print(token_span_end)
#         # print(entity_end)

#         # this covers the padded tokens at the end, if any
#         if (token_span_start, token_span_end) == (0,0):
#             if entity_vocab:
#                 chunk_entity_ids.append(-1)

#             chunk_bio_tags.append(2)
            

#         # first we check if the token is in the span and we haven't used a B tag already
#         elif (token_span_start >= entity_start) and (token_span_end <= entity_end) and (beginning_flag == False):
#             if entity_vocab:
#                 chunk_entity_ids.append([entity[4]])

#             chunk_bio_tags.append(0)
#             beginning_flag = True
#             in_span = True
#         # if we get here, we are already in a span, so we want to use an I tag
#         elif (token_span_start >= entity_start) and (token_span_end <= entity_end):
#             if entity_vocab:
#                 chunk_entity_ids.append(entity[4])

#             chunk_bio_tags.append(1)
#             in_span = True
#         # otherwise we're not in a span, and we must either have a Beginning or Outside tag
#         # we haven't changed our entity yet so it could be the next entity
#         else:
#             # if we're at the end of the entity list, then we must have an Outside tag
#             if (curr_entity_idx + 1) == len(data[test_pmid]['entities']):
#                 in_span = False
#                 beginning_flag = False
#                 if entity_vocab:
#                     chunk_entity_ids.append(-1)

#                 chunk_bio_tags.append(2)
                
            
#             else:
#                 # we'll check if we match the next entity
#                 next_entity = data[test_pmid]['entities'][curr_entity_idx + 1]
#                 next_entity_start = next_entity[0]
#                 next_entity_end = next_entity[1]
#                 # if we do, we add a Beginning tag and increment the entity counter
#                 if (token_span_start >= next_entity_start) and (token_span_end <= next_entity_end):
#                     chunk_bio_tags.append(0)
#                     if entity_vocab:
#                         chunk_entity_ids.append(next_entity[4])

#                     curr_entity_idx += 1
#                     in_span = True
#                     beginning_flag = True
                    
#                 # otherwise we don't and have an Outside Tag
#                 else:
#                     in_span = False
#                     beginning_flag = False
#                     if entity_vocab:
#                         chunk_entity_ids.append(-1)

#                     chunk_bio_tags.append(2)
        

#     # end with an Outside (O) tag for the [SEP] token
#     # chunk_bio_tags.append(2)
#     print(chunk_entity_ids)
#     print(chunk_bio_tags )
#     for idx, entity in enumerate(data[test_pmid]['entities']):
#         print(entity)
#         if idx ==50:
#             break

#     print(tokenizer.convert_ids_to_tokens(input_ids[:50]))
#     print("\n\n")
#     # print(tokenizer.vocab_size)
#     # print(output)
    

#     entity_ids.append(chunk_entity_ids)
#     bio_tags.append(chunk_bio_tags)
    
#     # assert sum([1 for val in chunk_bio_tags if val != 2]) == np.sum(np.array(chunk_entity_ids) >= 0), f"There was an issue in the BIO tagging: {[1 for val in chunk_bio_tags if val == 0]},{np.unique(chunk_entity_ids)}"

#     assert sum([1 for val in chunk_bio_tags if val != 2]) == sum([1 for ent in chunk_entity_ids if ent != -1]), f"There was an issue in the BIO tagging: {[1 for val in chunk_bio_tags if val == 0]},{np.unique(chunk_entity_ids)}"

####################################################################
# print(output_bio_tags)
# print(len(output.offset_mapping[1:-1]))
# for idx, entity in enumerate(data[test_pmid]['entities']):
#     print(entity)
#     output.offset_mapping
#     if idx ==8:
#         break

# print(tokenizer.convert_ids_to_tokens(output.input_ids[:50]))
# print(tokenizer.vocab_size)
# print(output)

