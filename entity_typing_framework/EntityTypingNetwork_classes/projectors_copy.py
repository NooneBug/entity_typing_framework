import json
from entity_typing_framework.utils.implemented_classes_lvl1 import IMPLEMENTED_CLASSES_LVL1
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import Sigmoid, ModuleDict, ReLU, Linear, Dropout, BatchNorm1d, Softmax
import torch
from copy import deepcopy


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
        elif activation_name == 'softmax':
            return Softmax(dim=1)
        elif activation_name == 'none':
            return None
        else:
            raise Exception('An unknown name (\'{}\')is given for activation, check the yaml or implement an activation that correspond to that name'.format(activation_name))

class Projector(LightningModule):
    '''

    A projector is a module that projects the encoded input in a joint space where the types are represented.

    The joint space has to be in accord with the :code:`TypeEncoder` module 

    Parameters:
        name:
            the name of the submodule, has to be specified in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.input_projector_params.name`

            to instance this projector insert the string :code:`Classifier` in the :code:`yaml` configuration file with key :code:`model.ET_Network_params.input_projector_params.name`

            this param is used by the :ref:`Entity Typing Network<EntityTypingNetwork>` to instance the correct submodule
        type2id:
            vocabulary that map each type name to an id. By default is created by the :code:`DatasetManager` and automatically passed through the :code:`MainModule`
        type_number:
            number of types in the dataset for this run, it is automatically extracted by the :doc:`DatasetManager<dataset_managers>` and automatically given in input to the Classifier by the :ref:`Entity Typing Network<EntityTypingNetwork>`
        input_dim:
            dimension of the vector inputed for the forward, it is automatically extracted from the :doc:`Encoder<encoders>` by the :ref:`Entity Typing Network<EntityTypingNetwork>`
        return_logits:
            if True, the projector acts like a classifier, so the type space is intended as a one hot space. 
            
            Thus the :code:`forward` method returns a value between 0 and 1 for each type in type2id following the behavior defined in :code:`classify()`.  
    '''
    def __init__(self, name, type2id, type_number, input_dim, return_logits = True, **kwargs):
        super().__init__()
        self.type_number = type_number
        self.input_dim = input_dim
        self.type2id = type2id
        self.return_logits = return_logits
    
    def project_input(self, input_representation):
        '''
        use this method to project the input_representation into a joint space where the also the types are represented 
        '''
        raise NotImplementedError

    def classify(self, projected_input):
        '''
        use this method to classify the projected_input if needed.
        '''
        raise NotImplementedError

    def forward(self, encoded_input):
        '''
        operates the forward pass of this submodule, projecting the encoded input in a joint space where labels are represented.
        if self.return_logits == True, the projected input is converted in a vector of confidence values (one for each type in the dataset)

        parameters:
            input_representation:
                output of the :doc:`Input Encoder<encoders>` with shape :code:`[input_dim, batch_size]`
        
        output:
            classification vector with shape :code:`[type_number, batch_size]`
        '''
        projected_input = self.project_input(input_representation=encoded_input)
        if self.return_logits:
            classifier_output = self.classify(projected_input=projected_input)
            return classifier_output
        else:
            return projected_input  
          
    def get_state_dict(self, smart_save=True):
        return self.state_dict()

class Classifier(Projector):
    '''
    Son of Projector, see :code:`Projector` documentation for basic parameters.

    Projector used as classification layer after the :ref:`Encoder<encoder>`. Predicts a vector with shape :code:`(type_number)` with values between 0 and 1.

    Parameters:
        layers_parameters:
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
    def classify(self, projected_input):
        return self.layers[str(len(self.layers)-1)](projected_input)

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


class ProjectorForIncrementalTraining(Projector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kwargs_pretraining = self.get_kwargs_pretrained_projector(**kwargs)
        self.pretrained_projector = self.get_class_for_pretrained_projector()(**kwargs_pretraining)

        kwargs_additional_projector = self.get_kwargs_incremental_projector(**kwargs)
        self.additional_projector = self.get_class_for_incremental_projector()(**kwargs_additional_projector)

        self.new_types = list(set(self.additional_projector.type2id.keys()) - set(self.pretrained_projector.type2id.keys()))

    def get_class_for_pretrained_projector(self):
        '''
        returns the class to instantiate the pretrained projector e.g. Classifier
        '''
        raise NotImplementedError


    def get_class_for_incremental_projector(self):
        '''
        returns the class to instantiate the incremental projector e.g. Classifier
        '''
        raise NotImplementedError
        
    def forward(self, input_representation):
        # project pretraining types
        pretrained_projected_representation = self.pretrained_projector.project_input(input_representation)
        
        # project incremental types
        incremental_projected_representation = self.additional_projector.project_input(input_representation)
        

        if self.return_logits:
            pretrained_output = self.pretrained_projector.classify(pretrained_projected_representation)
            incremental_output = self.additional_projector.classify(incremental_projected_representation)
            return pretrained_output, incremental_output
        else:
            return pretrained_projected_representation, incremental_projected_representation
        
    def get_kwargs_pretrained_projector(self, **kwargs):
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

    
    def get_new_type_number(self, **kwargs):
        new_type_number = len(kwargs['type2id']) - kwargs['type_number']
        
        return new_type_number

    def get_kwargs_incremental_projector(self, **kwargs):
        '''
        the composition of kwargs depends on the projector to instantiate (declared into :code:`self.get_class_for_incremental_projector()` method)
        '''
        raise NotImplementedError

    def freeze_pretrained(self):
        self.pretrained_projector.freeze()
        self.additional_projector.unfreeze()
    
    def copy_pretrained_parameters_into_incremental_module(self):
        raise NotImplementedError
        

class ClassifierForIncrementalTraining(ProjectorForIncrementalTraining):

    def __init__(self, layers_parameters, **kwargs):
       super().__init__(layers_parameters = layers_parameters, **kwargs)

    def get_class_for_pretrained_projector(self):
       return Classifier

    def get_class_for_incremental_projector(self):
        return Classifier

    def get_kwargs_incremental_projector(self, **kwargs):

        new_type_number = self.get_new_type_number(**kwargs)

        kwargs_additional_classifiers = deepcopy(kwargs)
        # prepare additional classifier with out_features set to new_type_number
        single_layers = sorted(kwargs_additional_classifiers['layers_parameters'].items())
        single_layers[-1][1]['out_features'] = new_type_number
        layers_parameters = {k: v for k, v in single_layers}
        kwargs_additional_classifiers['type_number'] = new_type_number
        kwargs_additional_classifiers['layers_parameters'] = layers_parameters
        return kwargs_additional_classifiers

    def copy_pretrained_parameters_into_incremental_module(self):
        # assuming that pretrained_projector and additional_projector have the same architecture
        # copy shared parameters
        for pretrained_l, incremental_l in zip(list(self.pretrained_projector.layers.values())[:-1], 
                                                list(self.additional_projector.layers.values())[:-1]):
            incremental_l.linear.weight = torch.nn.Parameter(pretrained_l.linear.weight.detach().clone())
            incremental_l.linear.bias = torch.nn.Parameter(pretrained_l.linear.bias.detach().clone())
        # init new parameters to better exploit the hierarchy: the weights of a logit of a new type are set to the values of the father's ones
        last_pretrained_layer = list(self.pretrained_projector.layers.values())[-1]
        last_incremental_layer = list(self.additional_projector.layers.values())[-1]
        for t in self.new_types:
            father = '/'.join(t.split('/')[:-1])
            idx_father = self.pretrained_projector.type2id[father]
            idx_t = self.additional_projector.type2id[t] - self.pretrained_projector.type_number
            last_incremental_layer.linear.weight.data[idx_t] = torch.nn.Parameter(last_pretrained_layer.linear.weight[idx_father].detach().clone())
            last_incremental_layer.linear.bias.data[idx_t] = torch.nn.Parameter(last_pretrained_layer.linear.bias[idx_father].detach().clone())

class ProjectorForCrossDatasetTraining(ProjectorForIncrementalTraining):
    # def __init__(self, **kwargs):
    #     kwargs_pretraining = kwargs['pretraining_input_projector_params']
    #     self.pretrained_projector = IMPLEMENTED_CLASSES_LVL1[kwargs_pretraining['name']](**kwargs_pretraining)

    #     kwargs_additional_projector = kwargs['incremental_input_projector_params']
    #     self.additional_projector = IMPLEMENTED_CLASSES_LVL1[kwargs_additional_projector['name']](**kwargs_additional_projector)

    #     self.new_types = list(set(self.additional_projector.type2id.keys()) - set(self.pretrained_projector.type2id.keys()))

    #     self.src2dst = self.read_src2dst(kwargs['src2dest_filepath'])

    def __init__(self, **kwargs):
       super().__init__(**kwargs)
       self.src2dst = self.read_src2dst(kwargs['src2dest_filepath'])
    
    def get_kwargs_pretrained_projector(self, **kwargs):
        return kwargs['pretraining_input_projector']

    def get_kwargs_incremental_projector(self, **kwargs):
        return kwargs['incremental_input_projector']
    
    def get_new_type_number(self, **kwargs):
        new_type_number = len(kwargs['type2id']) - kwargs['type_number']
        
        return new_type_number

        

    def read_src2dst(self, filepath):
        return json.loads(open(filepath, 'r'))

    def forward(self, input_representation):
        # return only incremental (dst dataset) types
        return super().forward(input_representation)[1]

class ClassifierForCrossDatasetTraining(ProjectorForCrossDatasetTraining):
    def get_class_for_pretrained_projector(self):
       return Classifier

    def get_class_for_incremental_projector(self):
        return Classifier

    def copy_pretrained_parameters_into_incremental_module(self):
        # init new parameters to better exploit the hierarchy: the weights of a logit of a type of the destination dataset is set to the value of the corresponding type of the source dataset
        last_pretrained_layer = list(self.pretrained_projector.layers.values())[-1]
        last_incremental_layer = list(self.additional_projector.layers.values())[-1]
        for t_src, t_dst in self.src2dst.items():
            idx_src = self.pretrained_projector.type2id[t_src]
            idx_dst = self.additional_projector.type2id[t_dst] - self.pretrained_projector.type_number
            last_incremental_layer.linear.weight.data[idx_dst] = torch.nn.Parameter(last_pretrained_layer.linear.weight[idx_src].detach().clone())
            last_incremental_layer.linear.bias.data[idx_dst] = torch.nn.Parameter(last_pretrained_layer.linear.bias[idx_src].detach().clone())
        # init dst subtypes with values of dst father types if there is a mapping of father, but there is no mapping of subtype
        for t in self.new_types:
            if t.split('/') > 2 and t not in self.src2dst.values():
                father = '/'.join(t.split('/')[:-1])
                subtype = t
                if father in self.src2dst.values():
                    idx_father = self.additional_projector.type2id[father]
                    idx_subtype = self.additional_projector.type2id[subtype]
                    last_incremental_layer.linear.weight.data[idx_subtype] = torch.nn.Parameter(last_pretrained_layer.linear.weight[idx_father].detach().clone())
                    last_incremental_layer.linear.bias.data[idx_subtype] = torch.nn.Parameter(last_pretrained_layer.linear.bias[idx_father].detach().clone()) 