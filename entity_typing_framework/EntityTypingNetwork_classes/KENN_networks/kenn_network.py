from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier
from pytorch_lightning import LightningModule
import torch
from torch.nn import Sigmoid
from tqdm import tqdm
import entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_utils as kenn_utils
import sys
sys.path.append('./')
from kenn.parsers import unary_parser

class KENNClassifier(LightningModule):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down', **kwargs):
    
    super().__init__()
    # classifier
    self.classifier = Classifier(**kwargs)

    if not clause_file_path:
      clause_file_path = 'kenn_tmp/clause_file_path.txt'
      id2type = {v: k for k, v in self.classifier.type2id.items()}
      # generate and save KENN clauses
      self.automatic_build_clauses(types_list = [id2type[idx] for idx in range(len(id2type))], clause_file_path=clause_file_path,
                                  learnable_clause_weight=learnable_clause_weight, clause_weight=clause_weight, kb_mode=kb_mode)
    
    # instance KENN layer by reading the clauses from the clause_file_path file 
    self.ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x, # linear activation
                          initial_clause_weight=clause_weight
                          )
    self.sig = Sigmoid()

  def forward(self, input_representation):
    prekenn = self.classifier(input_representation=input_representation)

    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(postkenn)
  
  # def forward(self, input_representation):
  #   projection_layers_output = self.project_input(input_representation)

  #   classification = self.classify(projection_layers_output)

  #   return self.sig(classification)

  # def project_input(self, input_representation):
  #   ...

  # def classify(self, projected_representation):
  #   ...


  def automatic_build_clauses(self, types_list, clause_file_path = None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down'):
    # generate and save KENN clauses
    cw = '_' if learnable_clause_weight else clause_weight
    kenn_utils.generate_constraints(types_list, kb_mode, clause_file_path, cw)

class KENNClassifierForIncrementalTraining(KENNClassifier):
  def __init__(self, clause_file_path=None, learnable_clause_weight=False, clause_weight=0.5, kb_mode='top_down', **kwargs):
    # extract info about pretraining types and incremental types
    type2id = kwargs['type2id']
    type_number_pretraining = kwargs['type_number']
    type_number_actual = len(type2id)
    new_type_number = type_number_actual - type_number_pretraining

    # remove new types from type2id to obtain pretraining_type2id and use it to reinstantiate the KnowledgeEnhancer
    # NOTE: it is assumed that new types are always provided at the end of the list
    type2id_pretraining = {k: v for k, v in list(type2id.items())[:-new_type_number]}
    kwargs_pretraining = {k:v for k,v in kwargs.items()}
    kwargs_pretraining['type2id'] = type2id_pretraining
    super().__init__(clause_file_path, learnable_clause_weight, clause_weight, kb_mode, **kwargs_pretraining)
    
    # prepare additional classifier with out_features set to new_type_number
    single_layers = sorted(kwargs['layers_parameters'].items())
    single_layers[-1][1]['out_features'] = new_type_number
    layers_parameters = {k: v for k, v in single_layers}
    kwargs_additional_classifiers = {k:v for k,v in kwargs.items()}
    kwargs_additional_classifiers['type_number'] = new_type_number
    kwargs_additional_classifiers['layers_parameters'] = layers_parameters
    self.additional_classifier = Classifier(**kwargs_additional_classifiers)
    
    # prepare additional Knowledge Enhancer
    all_types = list(type2id.keys())
    new_types = all_types[-new_type_number:]
    clause_file_path = 'kenn_tmp/incremental_clause_file_path.txt'
    cw = '_' if learnable_clause_weight else clause_weight
    kenn_utils.generate_constraints_incremental(all_types=all_types,
                                                new_types=new_types,
                                                filepath=clause_file_path,
                                                weight=cw)
    self.additional_ke = unary_parser(knowledge_file=clause_file_path,
                          activation=lambda x: x, # linear activation
                          initial_clause_weight=clause_weight
                          )

  def forward(self, input_representation):
    # predict pretraining types
    prekenn_pretrain = self.classifier(input_representation)
    postkenn_pretrain = self.ke(prekenn_pretrain)[0]
    
    # predict incremental types
    prekenn_incremental = self.additional_classifier(input_representation)
    prekenn_all_types = torch.concat((postkenn_pretrain, prekenn_incremental), dim=1)
    postkenn_incremental = self.additional_ke(prekenn_all_types)[0][:,-self.additional_classifier.type_number:]
    
    # assemble final prediction
    postkenn_all_types = torch.concat((postkenn_pretrain, postkenn_incremental), dim=1)

    return self.sig(postkenn_all_types)


  def freeze_pretraining(self):
    self.classifier.freeze()
    self.ke.freeze()
    # self.ke.knowledge_enhancer.clause_enhancers[indici_nuove_clausole].unfreeze() ?



class KENNClassifierMultiloss(KENNClassifier):
  def __init__(self, clause_file_path=None, **kwargs):
    super().__init__(clause_file_path, **kwargs)

  

  def forward(self, input_representation):
    prekenn = self.classifier(input_representation=input_representation)

    postkenn = self.ke(prekenn)[0]
    # self.ke(prekenn)[0] -> output
    # self.ke(prekenn)[1] -> deltas_list
    return self.sig(prekenn), self.sig(postkenn)

