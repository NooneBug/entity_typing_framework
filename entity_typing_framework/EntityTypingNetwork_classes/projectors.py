from typing import Optional
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_classes import CenterSigmoidBoxTensor
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import Sigmoid, ModuleDict, ReLU, Linear, Dropout, BatchNorm1d
import torch
from copy import deepcopy
import torch.nn as nn
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_classes import CenterSigmoidBoxTensor, BoxTensor, log1mexp
from typing import Optional, Tuple
import torch.nn.functional as F

class Layer(LightningModule):
    '''
    Fully Connected Layer with activation function, with parametrization managed through the :code:`yaml` configuration file under the key :code:`model.ET_Network_params.input_projector_params.layer_id`

    Each layer has an incremental id specified in the :code:`yaml` configuration file, which works as index in the dictionary (see the example configuration files)

    Parameters:
        in_features:
            dimension of the input features of the fully connected layer
        out_features:
            dimension of the output features of the fully connected layer
        activation:
            the activation function to use; supported activations are :code:`relu` and :code:`sigmoid`
        use_dropout:
            if use the dropout or not
        dropout_p:
            probability of dropout
        use_batch_norm:
            if use the batch normalization or not
    '''
    def __init__(self, in_features, out_features, activation = 'relu', use_dropout = True, dropout_p = .1, use_batch_norm = False) -> None:
        super().__init__()
        
        self.linear = Linear(in_features, out_features)

        self.activation = self.instance_activation(activation)
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        if self.use_dropout:
            self.dropout = Dropout(p = dropout_p)
        if self.use_batch_norm:
            self.batch_norm = BatchNorm1d(num_features=out_features)

    def forward(self, hidden_representation):
        '''
        Performs the forward pass for the fully connected layer.

        Parameters:
            hidden representation: 
                a tensor with shape :code:`[in_features, batch_size]`
        
        Output:
            output of the forward pass and the activation with shape :code:`[out_features, batch_size]`
        '''
        
        h = self.linear(hidden_representation)

        if self.activation:
            h = self.activation(h)

        if self.use_batch_norm:
            h = self.batch_norm(h)

        if self.use_dropout:
            h = self.dropout(h)            

        return h
    
    def instance_activation(self, activation_name):
        '''
        instances the activation function. This procedure is driven by the :code:`yaml` configuration file

        parameters:
            activation name:
                name of the activation function to use, specified in the key: :code:`model.ET_Network_params.input_projector_params.layer_id.activation` of the :code:`yaml` configuration file

                supported value : :code:`['relu', 'sigmoid']`

        '''
        if activation_name == 'relu':
            return ReLU()
        elif activation_name == 'sigmoid':
            return Sigmoid()
        elif activation_name == 'none':
            return None
        else:
            raise Exception('An unknown name (\'{}\')is given for activation, check the yaml or implement an activation that correspond to that name'.format(activation_name))

class Projector(LightningModule):
    def __init__(self, name, type2id, type_number, input_dim, **kwargs):
        super().__init__()
        self.type_number = type_number
        self.input_dim = input_dim
        self.type2id = type2id
    
class Classifier(Projector):
    '''
    Projector used as classification layer after the :ref:`Encoder<encoder>`. Predicts a vector with shape :code:`(type_number)` with values between 0 and 1.

    Parameters:
        name:
            the name of the submodule, has to be specified in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.input_projector_params.name`

            to instance this projector insert the string :code:`Classifier` in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.input_projector_params.name`

            this param is used by the :ref:`Entity Typing Network<EntityTypingNetwork>` to instance the correct submodule
        type_number:
            number of types in the dataset for this run, it is automatically extracted by the :doc:`DatasetManager<dataset_managers>` and automatically given in input to the Classifier by the :ref:`Entity Typing Network<EntityTypingNetwork>`
        input_dim:
            dimension of the vector inputed for the forward, it is automatically extracted from the :doc:`Encoder<encoders>` by the :ref:`Entity Typing Network<EntityTypingNetwork>`
        parameters:
            dictionary of parameters to instantiate different :code:`Layer` objects.

            the values for this parameter have to be specified in a :code:`yaml` dictionary in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.input_projector_params.layers_parameters`

            see the documentation of :code:`Layer` for the format of these parameters 
    '''

    def __init__(self, layers_parameters, **kwargs):
        super().__init__(**kwargs)
        self.layers_parameters = layers_parameters
        
        self.add_parameters()
        self.check_parameters()
        
        self.layers = ModuleDict({layer_name: Layer(**layer_parameters) for layer_name, layer_parameters in self.layers_parameters.items()})
    
    def forward(self, input_representation):
        '''
        operates the forward pass of this submodule, projecting the encoded input in a vector of confidence values (one for each type in the dataset)

        parameters:
            input_representation:
                output of the :doc:`Input Encoder<encoders>` with shape :code:`[input_dim, batch_size]`
        
        output:
            classification vector with shape :code:`[type_number, batch_size]`
        '''
        projection_layers_output = self.project_input(input_representation)
        classifier_output = self.classify(projection_layers_output)
        return classifier_output

    # TODO: documentation
    def project_input(self, input_representation):
        # iteratively forward except the classification layer
        for i in range(len(self.layers_parameters)-1):
            if i == 0:
                h = self.layers[str(i)](input_representation)
            else:
                h = self.layers[str(i)](h)

        return h
    
    # TODO: documentation
    def classify(self, projected_representation):
        return self.layers[str(len(self.layers)-1)](projected_representation)

    def add_parameters(self):
        '''
        adds the default parameters if are not specified into the :code:`yaml` configuration file under the key :code:`model.ET_Network_params.input_projector_params.layers_parameters`

        The default values are: 
            - if input features of the 0th projection layer are not specified or it is specified with the string :code:`encoder_dim`, the value :code:`input_dim` is inserted by default
            - if input features of a projection layer is specified with the string :code:`previous_out_features` the value :code:`out_features` of the previous layer is inserted
            - if output features of the last proection layer are not specificied or it is specified with the string :code:`type_number`: the value :code:`type_number` is inserted by default
        '''
        if 'in_features' not in self.layers_parameters['0']:
            self.layers_parameters['0']['in_features'] = self.input_dim
        
        if self.layers_parameters['0']['in_features'] == 'encoder_dim':
            self.layers_parameters['0']['in_features'] = self.input_dim
        
        if 'out_features' not in self.layers_parameters[str(len(self.layers_parameters) - 1)]:
            self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.type_number        
        
        if self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] == 'type_number':
            self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.type_number

        previous_out_features = self.input_dim
        for k in self.layers_parameters:
            if self.layers_parameters[k]['out_features'] == 'in_features':
                self.layers_parameters[k]['out_features'] = self.layers_parameters[k]['in_features']
            if self.layers_parameters[k]['in_features'] == 'previous_out_features':
                self.layers_parameters[k]['in_features'] = previous_out_features
            previous_out_features = self.layers_parameters[k]['out_features']



    def check_parameters(self):
        '''
        Check the parameters values and raises exceptions. Ensure that a classic classification can be obtained.
        '''
        if self.input_dim != self.layers_parameters['0']['in_features']:
            raise Exception('Encoder\'s output dimension ({}) and projector\'s input dimension ({}) has to have the same value ({}). Check the yaml'.format(self.input_dim, self.layers_parameters['0']['in_features'], self.input_dim))
        
        if self.type_number != self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features']:
            raise Exception('Types\' number ({}) and projector\'s last layer output dimension ({}) has to have the same value ({}). Check the yaml'.format(self.type_number, self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'], self.type_number))
    
    def get_state_dict(self, smart_save=True):
        return self.state_dict()


class ClassifierForIncrementalTraining(Classifier):
    def __init__(self, **kwargs):
        kwargs_pretraining = self.get_kwargs_pretraining(**kwargs)
        super().__init__(**kwargs_pretraining)
        kwargs_additional_classifier = self.get_kwargs_additional_classifier(**kwargs)
        self.additional_classifier = Classifier(**kwargs_additional_classifier)

    def forward(self, input_representation):
        # predict pretraining types
        pretrain_output = super().forward(input_representation)
        
        # predict incremental types
        incremental_projected_representation = self.project_input(input_representation)
        incremental_output = self.additional_classifier.classify(incremental_projected_representation)
        
        # assemble final prediction
        output_all_types = torch.concat((pretrain_output, incremental_output), dim=1)

        return output_all_types

    def get_kwargs_pretraining(self, **kwargs):
        # extract info about pretraining types and incremental types
        kwargs_pretraining = deepcopy(kwargs)
        type2id = kwargs_pretraining['type2id']
        type_number_pretraining = kwargs_pretraining['type_number']
        type_number_actual = len(type2id)
        new_type_number = type_number_actual - type_number_pretraining
        # remove new types from type2id
        # NOTE: it is assumed that new types are always provided at the end of the list
        type2id_pretraining = {k: v for k, v in list(type2id.items())[:-new_type_number]}
        kwargs_pretraining['type2id'] = type2id_pretraining
        return kwargs_pretraining

    def get_kwargs_additional_classifier(self, **kwargs):
        # prepare additional classifier with out_features set to new_type_number
        kwargs_additional_classifiers = deepcopy(kwargs)
        new_type_number = len(kwargs_additional_classifiers['type2id']) - kwargs_additional_classifiers['type_number']
        single_layers = sorted(kwargs_additional_classifiers['layers_parameters'].items())
        single_layers[-1][1]['out_features'] = new_type_number
        # layers_parameters = {k: v for k, v in single_layers}
        layers_parameters = {'0' : single_layers[-1][1]}
        kwargs_additional_classifiers['type_number'] = new_type_number
        kwargs_additional_classifiers['layers_parameters'] = layers_parameters
        return kwargs_additional_classifiers

    def freeze_pretraining(self):
        self.freeze()
        self.additional_classifier.unfreeze()

class BoxEmbeddingProjector(Projector):
    def __init__(self, type_number, input_dim, projection_network_params, box_decoder_params, box_embeddings_dimension=109, **kwargs) -> None:
        super().__init__(input_dim=input_dim, type_number=type_number, **kwargs)
        
        self.box_embedding_dimension = box_embeddings_dimension
        self.projection_network = HighwayNetwork(input_dim = input_dim,
        output_dim=self.box_embedding_dimension * 2,
        **projection_network_params)
        self.mc_box = CenterSigmoidBoxTensor
        self.box_decoder = BoxDecoder(num_embeddings = type_number, embedding_dim = 109, **box_decoder_params)

    def forward(self, encoded_input):
        # use HigwayNetwork to project the encoded input to the joint space with Types' Box Embeddings  
        projected_input = self.projection_network(encoded_input)

        # assuming that Boxes are CenterSigmoidBoxTensor, split the projected_input (a single tensor for each batch element) into two tensors for each batch elements to represent the box 
        mention_context_rep = self.mc_box.from_split(projected_input)

        log_probs = self.box_decoder(mention_context_rep)

        return mention_context_rep, log_probs
    
    def get_state_dict(self, smart_save=True):
        return self.state_dict()

box_types = {
'CenterSigmoidBoxTensor': CenterSigmoidBoxTensor
}

class BoxDecoder(nn.Module):
  def __init__(self,
               num_embeddings: int,
               embedding_dim: int = 109,
               box_type: str = 'CenterSigmoidBoxTensor',
               padding_idx: Optional[int] = None,
               max_norm: Optional[float] = None,
               norm_type: float = 2.,
               scale_grad_by_freq: bool = False,
               sparse: bool = False,
               _weight: Optional[torch.Tensor] = None,
               init_interval_delta: float = 0.5,
               init_interval_center: float = 0.01,
               inv_softplus_temp: float = 1.,
               softplus_scale: float = 1.,
               n_negatives: int = 0,
               neg_temp: float = 0.,
               box_offset: float = 0.5,
               pretrained_box: Optional[torch.Tensor] = None,
               use_gumbel_baysian: bool = False,
               gumbel_beta: float = 1.0):
    super(BoxDecoder, self).__init__()

    self.num_embeddings = num_embeddings
    self.box_embedding_dim = embedding_dim
    self.box_type = box_type
    try:
      self.box = box_types[box_type]
    except KeyError as ke:
      raise ValueError("Invalid box type {}".format(box_type)) from ke
    self.box_offset = box_offset  # Used for constant tensor
    self.init_interval_delta = init_interval_delta
    self.init_interval_center = init_interval_center
    self.inv_softplus_temp = inv_softplus_temp
    self.softplus_scale = softplus_scale
    self.n_negatives = n_negatives
    self.neg_temp = neg_temp
    self.use_gumbel_baysian = use_gumbel_baysian
    self.gumbel_beta = gumbel_beta
    self.box_embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim * 2,
                                       padding_idx=padding_idx,
                                       max_norm=max_norm,
                                       norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       sparse=sparse,
                                       _weight=_weight)

    self.euler_gamma = 0.57721566490153286060

    if pretrained_box is not None:
      print('Init box emb with pretrained boxes.')
      print(self.box_embeddings.weight)
      self.box_embeddings.weight = nn.Parameter(pretrained_box)
      print(self.box_embeddings.weight)
  
  def _compute_hard_min_max(
    box1: BoxTensor,
    box2: BoxTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns min and max points."""
    min_point = torch.max(box1.z, box2.z)
    max_point = torch.min(box1.Z, box2.Z)
    return min_point, max_point
    
  def _compute_gumbel_min_max(
    box1: BoxTensor,
    box2: BoxTensor,
    gumbel_beta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns min and max points."""
    min_point = torch.stack([box1.z, box2.z])
    min_point = torch.max(
        gumbel_beta * torch.logsumexp(min_point / gumbel_beta, 0),
        torch.max(min_point, 0)[0])

    max_point = torch.stack([box1.Z, box2.Z])
    max_point = torch.min(
        -gumbel_beta * torch.logsumexp(-max_point / gumbel_beta, 0),
        torch.min(max_point, 0)[0])
    return min_point, max_point
  
  def init_weights(self):
    print('before', self.box_embeddings.weight)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., :self.box_embedding_dim],
      -self.init_interval_center, self.init_interval_center)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., self.box_embedding_dim:],
      self.init_interval_delta, self.init_interval_delta)
    print('after', self.box_embeddings.weight)

  def log_soft_volume(
    self,
    z: torch.Tensor,
    Z: torch.Tensor,
    temp: float = 1.,
    scale: float = 1.,
    gumbel_beta: float = 0.) -> torch.Tensor:
    eps = torch.finfo(z.dtype).tiny  # type: ignore

    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale

    if gumbel_beta <= 0.:
      return (torch.sum(
        torch.log(F.softplus(Z - z, beta=temp).clamp_min(eps)),
        dim=-1) + torch.log(s)
              )  # need this eps to that the derivative of log does not blow
    else:
      return (torch.sum(
        torch.log(
          F.softplus(Z - z - 2 * self.euler_gamma * gumbel_beta, beta=temp).clamp_min(
          # F.softplus(((Z - z) / gumbel_beta) - 2 * self.euler_gamma, beta=temp).clamp_min(
            eps)),
          dim=-1) + torch.log(s))

  def type_box_volume(self) -> torch.Tensor:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
    if self.box_type == 'ConstantBoxTensor':
      type_box = self.box.from_split(emb, self.box_offset)
    else:
      type_box = self.box.from_split(emb)

    vol = self.log_soft_volume(type_box.z,
                               type_box.Z,
                               temp=self.inv_softplus_temp,
                               scale=self.softplus_scale,
                               gumbel_beta=self.gumbel_beta)
    return vol

  def get_pairwise_conditional_prob(self,
                                    type_x_ids: torch.Tensor,
                                    type_y_ids: torch.Tensor) -> torch.Tensor:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
    type_x = emb[type_x_ids]
    type_y = emb[type_y_ids]
    type_x_box = self.box.from_split(type_x)
    type_y_box = self.box.from_split(type_y)

    # Compute intersection volume
    if self.use_gumbel_baysian:
      # Gumbel intersection
      min_point, max_point = self._compute_gumbel_min_max(type_x_box,
                                                     type_y_box,
                                                     self.gumbel_beta)
    else:
      min_point, max_point = self._compute_hard_min_max(type_x_box, type_y_box)

    intersection_vol = self.log_soft_volume(min_point,
                                            max_point,
                                            temp=self.inv_softplus_temp,
                                            scale=self.softplus_scale,
                                            gumbel_beta=self.gumbel_beta)
    # Compute y volume here
    y_vol = self.log_soft_volume(type_y_box.z,
                                 type_y_box.Z,
                                 temp=self.inv_softplus_temp,
                                 scale=self.softplus_scale,
                                 gumbel_beta=self.gumbel_beta)

    # Need to be careful about numerical issues
    conditional_prob = intersection_vol - y_vol
    return torch.cat([conditional_prob.unsqueeze(-1),
                      log1mexp(conditional_prob).unsqueeze(-1)],
                     dim=-1)


  def forward(
    self,
    mc_box: torch.Tensor,
    # targets: Optional[torch.Tensor] = None,
    is_training: bool = True,
    batch_num: Optional[int] = None
  ) -> Tuple[torch.Tensor, None]:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim

    if self.box_type == 'ConstantBoxTensor':
      type_box = self.box.from_split(emb, self.box_offset)
    else:
      type_box = self.box.from_split(emb)

    # Get intersection
    batch_size = mc_box.data.size()[0]
    # Expand both mention&context and type boxes to the shape of batch_size x
    # num_types x box_embedding_dim. (torch.expand doesn't use extra memory.)
    if self.use_gumbel_baysian:  # Gumbel box
      min_point = torch.stack(
        [mc_box.z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
         type_box.z.unsqueeze(0).expand(batch_size, -1, -1)])
      min_point = torch.max(
        self.gumbel_beta * torch.logsumexp(min_point / self.gumbel_beta, 0),
        torch.max(min_point, 0)[0])

      max_point = torch.stack([
        mc_box.Z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
        type_box.Z.unsqueeze(0).expand(batch_size, -1, -1)])
      max_point = torch.min(
        -self.gumbel_beta * torch.logsumexp(-max_point / self.gumbel_beta, 0),
        torch.min(max_point, 0)[0])

    else:
      min_point = torch.max(
        torch.stack([
          mc_box.z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
          type_box.z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

      max_point = torch.min(
        torch.stack([
          mc_box.Z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
          type_box.Z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

    # Get soft volume
    # batch_size x num types
    # Compute the volume of the intersection
    vol1 = self.log_soft_volume(min_point,
                                max_point,
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta)

    # Compute  the volume of the mention&context box
    vol2 = self.log_soft_volume(mc_box.z,
                                mc_box.Z,
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta)

    # Returns log probs
    log_probs = vol1 - vol2.unsqueeze(-1)

    # Clip values > 1. for numerical stability.
    if (log_probs > 0.0).any():
      print("WARNING: Clipping log probability since it's grater than 0.")
      log_probs[log_probs > 0.0] = 0.0

    # if is_training and targets is not None and self.n_negatives > 0:
    #   pos_idx = torch.where(targets.sum(dim=0) > 0.)[0]
    #   neg_idx = torch.where(targets.sum(dim=0) == 0.)[0]

    #   if self.n_negatives < neg_idx.size()[0]:
    #     neg_idx = neg_idx[torch.randperm(len(neg_idx))[:self.n_negatives]]
    #     log_probs_pos = log_probs[:, pos_idx]
    #     log_probs_neg = log_probs[:, neg_idx]
    #     _log_probs = torch.cat([log_probs_pos, log_probs_neg], dim=-1)
    #     _targets = torch.cat([targets[:, pos_idx], targets[:, neg_idx]], dim=-1)
    #     _weights = None
    #     if self.neg_temp > 0.0:
    #       _neg_logits = log_probs_neg - log1mexp(log_probs_neg)
    #       _neg_weights = F.softmax(_neg_logits * self.neg_temp, dim=-1)
    #       _pos_weights = torch.ones_like(log_probs_pos,
    #                                      device=self.box_embeddings.weight.device)
    #       _weights = torch.cat([_pos_weights, _neg_weights], dim=-1)
    #     return _log_probs, _weights, _targets
    #   else:
    #     return log_probs, None, targets
    # elif is_training and targets is not None and self.n_negatives <= 0:
    #   return log_probs, None, targets
    # else:
    #   return log_probs, None, targets
    
    return log_probs

  def get_state_dict(self, smart_save=True):
        return self.state_dict()



class HighwayNetwork(LightningModule):
  def __init__(self,
               name,
               input_dim: int,
               output_dim: int,
               n_layers: int = 2,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](inputs))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)

