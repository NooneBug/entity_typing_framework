from copy import deepcopy
from entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_network import KENNModule
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector import BoxEmbeddingIncrementalProjector, BoxEmbeddingProjector
from entity_typing_framework.EntityTypingNetwork_classes.projectors import ProjectorForIncrementalTraining
import torch
import entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_utils as kenn_utils
from torch.nn import Sigmoid
from kenn.parsers import unary_parser

# util
def inverse_sigmoid(x):
    # from https://discuss.pytorch.org/t/inverse-of-sigmoid-in-pytorch/14215
    return torch.log(x/(1 - x))

class BoxEmbeddingKENNProjector(KENNModule):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down',**kwargs):
    super().__init__(clause_file_path, learnable_clause_weight, clause_weight, kb_mode, **kwargs)
    # classifier
    self.projector = BoxEmbeddingProjector(**kwargs)
    self.sigmoid = torch.nn.Sigmoid()

  def project_input(self, input_representation):
    return self.projector.project_input(input_representation)
  
  def classify(self, projected_input):
    return self.projector.classify(projected_input)

  def forward(self, input_representation):
    _, prekenn_log_prob = self.projector(encoded_input=input_representation)
    
    # change domain of prekenn_log_prob to apply KENN enhancement
    prob = torch.exp(prekenn_log_prob)
    prekenn = inverse_sigmoid(prob) 
    # compute postkenn scores
    postkenn = self.apply_knowledge_enhancement(prekenn)

    # change postkenn value to a log probability 
    postkenn_log_prob = torch.log(self.sigmoid(postkenn))

    return prekenn_log_prob, postkenn_log_prob

# class BoxEmbeddingKENNProjectorForIncrementalTraining(BoxEmbeddingIncrementalProjector):
#   def get_class_for_pretrained_projector(self):
#     return BoxEmbeddingProjector

#   def get_class_for_incremental_projector(self):
#     return BoxEmbeddingKENNProjector

#   def __init__(self, clause_file_path=None, learnable_clause_weight=False, clause_weight=0.5, kb_mode='top_down', **kwargs):
#     super().__init__(clause_file_path=clause_file_path, learnable_clause_weight=learnable_clause_weight, clause_weight=clause_weight, kb_mode=kb_mode, **kwargs)
    
#   def get_kwargs_incremental_projector(self, **kwargs):
#     # get kwargs of father class
#     father_kwargs = super().get_kwargs_incremental_projector(**kwargs)
    
#     new_type_number = self.get_new_type_number(**kwargs)

#     # get clauses only for incremental types
#     type2id = father_kwargs['type2id']
#     new_type_number = len(father_kwargs['type2id']) - kwargs['type_number']
#     all_types = list(type2id.keys())
#     new_types = all_types[-new_type_number:]
#     # TODO: modify next three lines to be consistent with newer classes
#     clause_file_path = f"kenn_tmp/{kwargs['kb_mode']}_incremental_clause_file.txt"
#     cw = '_' if kwargs['learnable_clause_weight'] else kwargs['clause_weight']
#     kenn_utils.generate_constraints_incremental(all_types=all_types,
#                                                 new_types=new_types,
#                                                 filepath=clause_file_path,
#                                                 weight=cw,
#                                                 mode=kwargs['kb_mode'])
    
#     # modify the kwargs to instantiate the correct ke by the super().__init__(call)
#     incremental_kwargs = deepcopy(father_kwargs)
#     incremental_kwargs['clause_file_path'] = clause_file_path
#     incremental_kwargs['type_number'] = new_type_number    

#     return incremental_kwargs
    

#   def forward(self, input_representation):
#     # predict pretraining types
#     pretrain_projected_input = self.pretrained_projector.project_input(input_representation)
#     prekenn_pretrain = self.pretrained_projector.classify(pretrain_projected_input)
#     postkenn_pretrain = self.pretrained_projector.apply_knowledge_enhancement(prekenn_pretrain)

#     # predict incremental types
#     prekenn_incremental = self.additional_projector.classifier(input_representation)

#     prekenn_all_types = torch.concat((postkenn_pretrain, prekenn_incremental), dim=1) # why postkenn_pretrain and not prekenn_pretrain? Because it is stacked...
    
#     postkenn_incremental = self.additional_projector.ke(prekenn_all_types)[0][:,-self.additional_projector.type_number:]

#     return (self.sig(postkenn_pretrain), self.sig(prekenn_incremental)), (self.sig(postkenn_pretrain), self.sig(postkenn_incremental))

#   # def copy_pretrained_parameters_into_incremental_module(self):
#   #     # assuming that pretrained_projector and additional_projector have the same architecture

#   #     # COPY BOX WEIGHTS
#   #     # copy nonlinear
#   #     for pretrained_l, incremental_l in zip(self.pretrained_projector.projection_network.nonlinear, 
#   #                                               self.additional_projector.projection_network.nonlinear):
#   #           incremental_l.weight = torch.nn.Parameter(pretrained_l.weight.detach().clone())
#   #           incremental_l.bias = torch.nn.Parameter(pretrained_l.bias.detach().clone())

#   #     # copy gate
#   #     for pretrained_l, incremental_l in zip(self.pretrained_projector.projection_network.gate, 
#   #                                               self.additional_projector.projection_network.gate):
#   #           incremental_l.weight = torch.nn.Parameter(pretrained_l.weight.detach().clone())
#   #           incremental_l.bias = torch.nn.Parameter(pretrained_l.bias.detach().clone())
      
#   #     # copy final_linear_layer
#   #     self.additional_projector.projection_network.final_linear_layer.weight = torch.nn.Parameter(self.pretrained_projector.projection_network.final_linear_layer.weight.detach().clone())
#   #     self.additional_projector.projection_network.final_linear_layer.bias = torch.nn.Parameter(self.pretrained_projector.projection_network.final_linear_layer.bias.detach().clone())

#   #     # copy BoxDecoder weights
#   #     # init new parameters to better exploit the hierarchy: the weights of the embedding of a new type are set to the values of the father's ones]
#   #     pretrained_box_decoder = self.pretrained_projector.box_decoder.box_embeddings
#   #     additional_box_decoder = self.additional_projector.box_decoder.box_embeddings
#   #     for t in self.new_types:
#   #         father = '/'.join(t.split('/')[:-1])
#   #         idx_father = self.pretrained_projector.type2id[father]
#   #         idx_t = self.additional_projector.type2id[t] - self.pretrained_projector.type_number
#   #         additional_box_decoder.weight.data[idx_t] = torch.nn.Parameter(pretrained_box_decoder.weight[idx_father].detach().clone())

class BoxEmbeddingKENNProjectorForIncrementalTraining(BoxEmbeddingIncrementalProjector):
  def get_class_for_pretrained_projector(self):
    return BoxEmbeddingProjector

  def get_class_for_incremental_projector(self):
    return BoxEmbeddingProjector

  def __init__(self, clause_file_path=None, learnable_clause_weight=False, clause_weight=0.5, kb_mode='top_down', **kwargs):
    super().__init__(**kwargs)
    self.sig = Sigmoid()
    if not clause_file_path:
      # generate and save KENN clauses
      clause_file_path = f'kenn_tmp/{kb_mode}.txt'
      id2type = {v: k for k, v in self.type2id.items()}
      types_list = [id2type[idx] for idx in range(len(id2type))]
      kenn_utils.generate_constraints(types_list, kb_mode, clause_file_path, learnable_clause_weight, clause_weight)
    
    # instance KENN layer by reading the clauses from the clause_file_path file 
    self.ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x, # linear activation
                          initial_clause_weight=clause_weight
                          )
    
  def get_kwargs_incremental_projector(self, **kwargs):
    # get kwargs of father class
    father_kwargs = super().get_kwargs_incremental_projector(**kwargs)
    
    # modify the kwargs to instantiate the correct ke by the super().__init__(call)
    incremental_kwargs = deepcopy(father_kwargs)
    incremental_kwargs['type_number'] = self.get_new_type_number(**kwargs)

    return incremental_kwargs
    

  def forward(self, input_representation):
    # predict pretraining types
    logprob_prekenn_pretrain = self.pretrained_projector(input_representation)[1] # [0] = mention_context_rep
    # convert log probs to probs
    prob_prekenn_pretrain = logprob_prekenn_pretrain.exp()
    # convert probs to preactivations of a sigmoid
    prekenn_pretrain = inverse_sigmoid(prob_prekenn_pretrain)
    
    # predict incremental types
    logprob_prekenn_incremental = self.additional_projector(input_representation)[1] # [0] = mention_context_rep
    # convert log probs to probs
    prob_prekenn_incremental = logprob_prekenn_incremental.exp()
    # convert probs to preactivations of a sigmoid
    prekenn_incremental = inverse_sigmoid(prob_prekenn_incremental)

    # concat prekenn preactivations
    prekenn_all_types = torch.concat((prekenn_pretrain, prekenn_incremental), dim=1)
    # apply kenn
    postkenn_all_types = self.ke(prekenn_all_types)[0]
    # keep only enhanced incremental predictions
    postkenn_incremental = postkenn_all_types[:,-self.additional_projector.type_number:]

    # clip presigmoid to avoid torch.log(torch.sig(X)) = -inf
    postkenn_incremental = torch.clamp(postkenn_incremental, min=-85)

    # convert output from preactivations to logprobs
    logprob_postkenn_incremental = torch.log(self.sig(postkenn_incremental))
    logprob_postkenn_pretrain = logprob_prekenn_pretrain # only for readability of return
    return (logprob_prekenn_pretrain, logprob_prekenn_incremental), (logprob_postkenn_pretrain, logprob_postkenn_incremental)