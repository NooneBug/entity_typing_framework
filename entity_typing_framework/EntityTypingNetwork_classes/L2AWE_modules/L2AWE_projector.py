from entity_typing_framework.EntityTypingNetwork_classes.projectors import Classifier, ClassifierForCrossDatasetTraining
import torch


class L2AWEClassifierForCrossDatasetTraining(ClassifierForCrossDatasetTraining):
  

  def forward(self, input_representation):
    # comput probability distribution over src types
    src_output = self.src_classifier(input_representation)
    # compose L2AWE input
    tgt_input_representation = torch.hstack((src_output, input_representation))
    # compute probability distribution over tgt types
    tgt_output = self.tgt_classifier(tgt_input_representation)
    return tgt_output
  
  def instance_tgt_classifier(self, layers_parameters, **kwargs):
    # dinamically set input shape according to:
    # " the input space is the concatenation of the probability distribution in the source schema and the
    # embedded representation related to the entity mention"
    layers_parameters['0']['in_features'] = kwargs['input_dim'] + len(self.src_type2id)
    return Classifier(layers_parameters, **kwargs)
  
  def copy_src_parameters_into_tgt_module(self):
    # the input shape is completely different, it would be a partial copy
    pass