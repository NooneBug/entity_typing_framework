def get_kwargs_pretraining(**kwargs):
    # extract info about pretraining types and incremental types
    type2id = kwargs['type2id']
    type_number_pretraining = kwargs['type_number']
    type_number_actual = len(type2id)
    new_type_number = type_number_actual - type_number_pretraining
    # remove new types from type2id
    # NOTE: it is assumed that new types are always provided at the end of the list
    type2id_pretraining = {k: v for k, v in list(type2id.items())[:-new_type_number]}
    kwargs_pretraining = {k:v for k,v in kwargs.items()}
    kwargs_pretraining['type2id'] = type2id_pretraining
    return kwargs_pretraining

def get_kwargs_additional_classifier(**kwargs):
    # prepare additional classifier with out_features set to new_type_number
    new_type_number = len(kwargs['type2id']) - kwargs['type_number']
    single_layers = sorted(kwargs['layers_parameters'].items())
    single_layers[-1][1]['out_features'] = new_type_number
    # layers_parameters = {k: v for k, v in single_layers}
    layers_parameters = {'0' : single_layers[-1][1]}
    kwargs_additional_classifiers = {k:v for k,v in kwargs.items()}
    kwargs_additional_classifiers['type_number'] = new_type_number
    kwargs_additional_classifiers['layers_parameters'] = layers_parameters
    return kwargs_additional_classifiers