from typing import Optional
from entity_typing_framework.EntityTypingNetwork_classes.box_embeddings_classes import CenterSigmoidBoxTensor
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import Sigmoid, ModuleDict, ReLU, Linear, Dropout, BatchNorm1d
import torch.nn as nn
import torch

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


class Classifier(LightningModule):
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

    def __init__(self, name, type_number, input_dim, layers_parameters):
        super().__init__()
        self.type_number = type_number
        self.input_dim = input_dim
        self.layers_parameters = layers_parameters

        self.add_parameters()
        self.check_parameters()
        
        self.layers = ModuleDict({layer_name: Layer(**layer_parameters) for layer_name, layer_parameters in self.layers_parameters.items()})
    
    def forward(self, input_representation):
        '''
        operates the forward pass of this submodule, proecting the encoded input in a vector of confidence values (one for each type in the dataset)

        parameters:
            input_representation:
                output of the :doc:`Input Encoder<encoders>` with shape :code:`[input_dim, batch_size]`
        
        output:
            classification vector with shape :code:`[type_number, batch_size]`
        '''
        for i in range(len(self.layers_parameters)):
            if i == 0:
                h = self.layers[str(i)](input_representation)
            else:
                h = self.layers[str(i)](h)

        return h

    def add_parameters(self):
        '''
        adds the default parameters if are not specified into the :code:`yaml` configuration file under the key :code:`model.ET_Network_params.input_projector_params.layers_parameters`

        The default values are: 
            - if input features of the 0th projection layer are not specified or it is specified the string :code:`encoder_dim`, the value :code:`input_dim` is inserted by default
            - if output features of the last proection layer are not specificied or it is specified the string :code:`type_number`: the value :code:`type_number` is inserted by default
        '''
        if 'in_features' not in self.layers_parameters['0']:
            self.layers_parameters['0']['in_features'] = self.input_dim
        
        if self.layers_parameters['0']['in_features'] == 'encoder_dim':
            self.layers_parameters['0']['in_features'] = self.input_dim
        
        if 'out_features' not in self.layers_parameters[str(len(self.layers_parameters) - 1)]:
            self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.type_number        
        
        if self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] == 'type_number':
            self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.type_number        


    def check_parameters(self):
        '''
        Check the parameters values and raises exceptions. Ensure that a classic classification can be obtained.
        '''
        if self.input_dim != self.layers_parameters['0']['in_features']:
            raise Exception('Encoder\'s output dimension ({}) and projector\'s input dimension ({}) has to have the same value ({}). Check the yaml'.format(self.input_dim, self.layers_parameters['0']['in_features'], self.input_dim))
        
        if self.type_number != self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features']:
            raise Exception('Types\' number ({}) and projector\'s last layer output dimension ({}) has to have the same value ({}). Check the yaml'.format(self.type_number, self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'], self.type_number))

class BoxEmbeddingProjector(LightningModule):
    def __init__(self, name, type_number, input_dim, projection_network_params, box_embeddings_dimension=109, **kwargs) -> None:
        super().__init__(**kwargs)
        self.type_number = type_number
        self.input_dim = input_dim
        self.box_embedding_dimension = box_embeddings_dimension
        self.projection_network = HighwayNetwork(input_dim = input_dim,
        output_dim=self.box_embedding_dimension * 2,
        **projection_network_params)
        self.mc_box = CenterSigmoidBoxTensor

    def forward(self, encoded_input):
        # use HigwayNetwork to project the encoded input to the joint space with Types' Box Embeddings  
        projected_input = self.projection_network(encoded_input)

        # assuming that Boxes are CenterSigmoidBoxTensor, split the projected_input (a single tensor for each batch element) into two tensors for each batch elements to represent the box 
        mention_context_rep = self.mc_box.from_split(projected_input)

        return mention_context_rep

class HighwayNetwork(nn.Module):
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

