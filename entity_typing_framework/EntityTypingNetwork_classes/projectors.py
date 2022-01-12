from pytorch_lightning.core.lightning import LightningModule
from torch.nn import Sigmoid, ModuleDict, ReLU, Linear, Dropout, BatchNorm1d
from torch.nn.modules import activation

class Layer(LightningModule):

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
        
        h = self.activation(self.linear(hidden_representation))

        if self.use_batch_norm:
            h = self.batch_norm(h)

        if self.use_dropout:
            h = self.dropout(h)            

        return h
    
    def instance_activation(self, activation_name):
        if activation_name == 'relu':
            return ReLU()
        elif activation_name == 'sigmoid':
            return Sigmoid()
        else:
            raise Exception('An unknown name (\'{}\')is given for activation, check the yaml or implement an activation that correspond to that name'.format(activation_name))


class Classifier(LightningModule):

    def __init__(self, type_number, input_dim, parameters):
        super().__init__()
        self.type_number = type_number
        self.input_dim = input_dim
        self.parameters = parameters
        self.layers_parameters = self.parameters['layers_parameters']

        self.add_parameters()
        self.check_parameters()
        
        self.layers = ModuleDict({layer_name: Layer(**layer_parameters) for layer_name, layer_parameters in self.layers_parameters.items()})
    
    def forward(self, input_representation):
        for i in range(len(self.layers_parameters)):
            if i == 0:
                h = self.layers[str(i)](input_representation)
            else:
                h = self.layers[str(i)](h)

        return h

    def add_parameters(self):
        if 'in_features' not in self.layers_parameters['0']:
            self.layers_parameters['0']['in_features'] = self.input_dim
        
        if self.layers_parameters['0']['in_features'] == 'encoder_dim':
            self.layers_parameters['0']['in_features'] = self.input_dim
        
        if 'out_features' not in self.layers_parameters[str(len(self.layers_parameters) - 1)]:
            self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.type_number        
        
        if self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] == 'type_number':
            self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'] = self.type_number        


    def check_parameters(self):
        if self.input_dim != self.layers_parameters['0']['in_features']:
            raise Exception('Encoder\'s output dimension ({}) and projector\'s input dimension ({}) has to have the same value ({}). Check the yaml'.format(self.input_dim, self.layers_parameters['0']['in_features'], self.input_dim))
        
        if self.type_number != self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features']:
            raise Exception('Types\' number ({}) and projector\'s last layer output dimension ({}) has to have the same value ({}). Check the yaml'.format(self.type_number, self.layers_parameters[str(len(self.layers_parameters) - 1)]['out_features'], self.type_number))
