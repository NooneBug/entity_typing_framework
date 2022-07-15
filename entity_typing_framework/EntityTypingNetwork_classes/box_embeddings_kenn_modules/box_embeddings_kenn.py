from entity_typing_framework.EntityTypingNetwork_classes.KENN_networks.kenn_network import KENNModule
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_modules.box_embedding_projector import BoxEmbeddingProjector
import torch

class BoxEmbeddingKENNProjector(KENNModule):
  def __init__(self, clause_file_path=None, learnable_clause_weight = False, clause_weight = 0.5, kb_mode = 'top_down',**kwargs):
    super().__init__(clause_file_path, learnable_clause_weight, clause_weight, kb_mode, **kwargs)
    # classifier
    self.classifier = BoxEmbeddingProjector(**kwargs)
    self.sigmoid = torch.nn.Sigmoid()

  def project_input(self, input_representation):
    return self.classifier.project_input(input_representation)
  
  def classify(self, projected_input):
    return self.classifier.classify(projected_input)

  def inverse_sigmoid(self, x):
    # from https://discuss.pytorch.org/t/inverse-of-sigmoid-in-pytorch/14215
    return torch.log(x/(1 - x))

  def forward(self, input_representation):
    _, prekenn_log_prob = self.classifier(encoded_input=input_representation)
    
    # change domain of prekenn_log_prob to apply KENN enhancement
    prob = torch.exp(prekenn_log_prob)
    prekenn = self.inverse_sigmoid(prob) 
    # compute postkenn scores
    postkenn = self.apply_knowledge_enhancement(prekenn)

    # change postkenn value to a log probability 
    postkenn_lop_prob = torch.log(self.sigmoid(postkenn))

    return prekenn_log_prob, postkenn_lop_prob