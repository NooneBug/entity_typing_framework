from copy import deepcopy
from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier, ClassifierForCrossDatasetTraining, ClassifierForIncrementalTraining, Projector
from pytorch_lightning import LightningModule
import torch
from torch.nn import Sigmoid, Linear
from tqdm import tqdm
import entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_utils as kenn_utils
import json

import sys
sys.path.append('./')
from kenn.parsers import unary_parser

class KENNModule(Projector):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down', **kwargs):
    super().__init__(**kwargs)

    self.type2id = kwargs['type2id']

    if not clause_file_path:
      clause_file_path = f'kenn_tmp/{kb_mode}_clause_file.txt'
      id2type = {v: k for k, v in self.type2id.items()}
      # generate and save KENN clauses
      self.automatic_build_clauses(types_list = [id2type[idx] for idx in range(len(id2type))], clause_file_path=clause_file_path,
                                  learnable_clause_weight=learnable_clause_weight, clause_weight=clause_weight, kb_mode=kb_mode)
    
    # instance KENN layer by reading the clauses from the clause_file_path file 
    self.ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x, # linear activation
                          initial_clause_weight=clause_weight
                          )
    
    self.kb_mode = kb_mode
  
  def apply_knowledge_enhancement(self, prekenn_input):
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.ke(prekenn_input)[0] if self.ke.knowledge_enhancer.clause_enhancers else prekenn_input
  
  def automatic_build_clauses(self, types_list, clause_file_path = None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down'):
    # generate and save KENN clauses
    cw = '_' if learnable_clause_weight else clause_weight
    kenn_utils.generate_constraints(types_list, kb_mode, clause_file_path, cw)

class KENNClassifier(KENNModule):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down',**kwargs):
    super().__init__(clause_file_path, learnable_clause_weight, clause_weight, kb_mode, **kwargs)
    # classifier
    self.classifier = Classifier(**kwargs)
    self.sig = Sigmoid()

  def project_input(self, input_representation):
    return self.classifier.project_input(input_representation)
  
  def classify(self, projected_input):
    return self.classifier.classify(projected_input)

  def forward(self, input_representation):
    prekenn = self.classifier(encoded_input=input_representation)
    postkenn = self.apply_knowledge_enhancement(prekenn)
    return self.sig(prekenn), self.sig(postkenn)


class KENNClassifierForIncrementalTraining(ClassifierForIncrementalTraining):

  def get_class_for_pretrained_projector(self):
    return KENNClassifier

  def get_class_for_incremental_projector(self):
    return KENNClassifier

  def __init__(self, clause_file_path=None, learnable_clause_weight=False, clause_weight=0.5, kb_mode='top_down', **kwargs):
    super().__init__(clause_file_path=clause_file_path, learnable_clause_weight=learnable_clause_weight, clause_weight=clause_weight, kb_mode=kb_mode, **kwargs)
    self.sig = Sigmoid()
    
  def get_kwargs_incremental_projector(self, **kwargs):
    # get kwargs of father class
    father_kwargs = super().get_kwargs_incremental_projector(**kwargs)
    
    # get clauses only for incremental types
    type2id = father_kwargs['type2id']
    new_type_number = len(father_kwargs['type2id']) - kwargs['type_number']
    all_types = list(type2id.keys())
    new_types = all_types[-new_type_number:]
    
    if 'clause_file_path' not in kwargs:
      clause_file_path = f"kenn_tmp/{kwargs['kb_mode']}_incremental_clause_file.txt"
    else:
      clause_file_path = kwargs['clause_file_path']
    cw = '_' if kwargs['learnable_clause_weight'] else kwargs['clause_weight']

    if kwargs['kb_mode'].lower() != 'none':
      kenn_utils.generate_constraints_incremental(all_types=all_types,
                                                  new_types=new_types,
                                                  filepath=clause_file_path,
                                                  weight=cw,
                                                  mode=kwargs['kb_mode'])
    
    # modify the kwargs to instantiate the correct ke by the super().__init__(call)
    incremental_kwargs = deepcopy(father_kwargs)
    incremental_kwargs['clause_file_path'] = clause_file_path

    return incremental_kwargs
    

  def forward(self, input_representation):
    # predict pretraining types
    pretrain_projected_input = self.pretrained_projector.project_input(input_representation)
    prekenn_pretrain = self.pretrained_projector.classify(pretrain_projected_input)

    # TODO: pretrained classifier has an unused KENN knowledge enhancer, remove it with refactoring on IncrementalModel
    # postkenn_pretrain = self.pretrained_projector.apply_knowledge_enhancement(prekenn_pretrain)

    postkenn_pretrain = prekenn_pretrain

    # predict incremental types
    prekenn_incremental = self.additional_projector.classifier(input_representation)

    prekenn_all_types = torch.concat((postkenn_pretrain, prekenn_incremental), dim=1) # why postkenn_pretrain and not prekenn_pretrain? Because it is stacked...
    
    postkenn_incremental = self.additional_projector.ke(prekenn_all_types)[0][:,-self.additional_projector.type_number:]

    return (self.sig(postkenn_pretrain), self.sig(prekenn_incremental)), (self.sig(postkenn_pretrain), self.sig(postkenn_incremental))

  # TODO: remove method!!! wrong initialization
  def copy_pretrained_parameters_into_incremental_module(self):
        # assuming that pretrained_projector and additional_projector have the same architecture
        for pretrained_l, incremental_l in zip(list(self.pretrained_projector.classifier.layers.values())[:-1], 
                                                list(self.additional_projector.classifier.layers.values())[:-1]):
            incremental_l.linear.weight = torch.nn.Parameter(pretrained_l.linear.weight.detach().clone())
            incremental_l.linear.bias = torch.nn.Parameter(pretrained_l.linear.bias.detach().clone())

class KENNClassifierForIncrementalTrainingOntonotes(KENNClassifierForIncrementalTraining):

  def get_class_for_pretrained_projector(self):
    return Classifier
  
  # TODO: remove method!!! wrong initialization
  def copy_pretrained_parameters_into_incremental_module(self):
        # assuming that pretrained_projector and additional_projector have the same architecture
        for pretrained_l, incremental_l in zip(list(self.pretrained_projector.layers.values())[:-1], 
                                                list(self.additional_projector.classifier.layers.values())[:-1]):
            incremental_l.linear.weight = torch.nn.Parameter(pretrained_l.linear.weight.detach().clone())
            incremental_l.linear.bias = torch.nn.Parameter(pretrained_l.linear.bias.detach().clone())

class KENNClassifierForCrossDatasetTraining(ClassifierForCrossDatasetTraining):
  def __init__(self, clause_file_path=None, clause_weight = 0.5, **kwargs):
    super().__init__(**kwargs)
    # set linear activation function instead of sigmoid to give the correct input to the ke
    last_idx = str(len(self.src_classifier.layers)-1)
    self.src_classifier.layers[last_idx].activation = self.src_classifier.layers[last_idx].instance_activation('none')
    self.sig = Sigmoid()

    if clause_file_path:
      print('KnowledgeEnhancer instantiated by using', clause_file_path)
      self.ke = unary_parser(knowledge_file=clause_file_path,
                      activation=lambda x: x, # linear activation
                      initial_clause_weight=clause_weight
                      )
    else:
      raise Exception('Clause file not provided!')

  def forward(self, input_representation):
    # compute prediction of src types to feed the ke
    # use classifier.classify since they have linear activation function on the last layer
    src_prekenn = self.src_classifier(input_representation)
    tgt_prekenn = self.tgt_classifier(input_representation)
    # build kenn input: NOTE: predicates = types(src) U types(tgt)
    stacked_prekenn = torch.hstack((src_prekenn, tgt_prekenn))
    stacked_postkenn = self.ke(stacked_prekenn)[0]
    # keep only tgt predictions
    tgt_postkenn = stacked_postkenn[:, -len(self.tgt_type2id):]
    return self.sig(tgt_prekenn), self.sig(tgt_postkenn)

  # def copy_src_parameters_into_tgt_module(self):
  #   src_layers = list(self.src_classifier.layers.values())
  #   tgt_layers = list(self.tgt_classifier.layers.values())

  #   # init new parameters to exploit previous knwoledge: initial layers
  #   for src_layer, tgt_layer in zip(src_layers[:-1], tgt_layers[:-1]):
  #       tgt_layer.linear.weight.data = torch.nn.Parameter(src_layer.linear.weight.detach().clone())
  #       tgt_layer.linear.bias.data = torch.nn.Parameter(src_layer.linear.bias.detach().clone())