import torch
from entity_typing_framework.utils.flat2hierarchy import get_type2id_original
import torch.nn.functional as F
from collections import defaultdict

class BaseInferenceManager():
    '''
    Inference manager to discretize the output of the network accordingly to a :code:`threshold`.
    If an output value is greater than :code:`threshold`, then assign 1; assign 0 otherwise.

    Parameters:
        threshold:
            threshold to use to discretize the output
        type2id:
            dict to map each type to an ID that corresponds to the output neuron of the network
        transitive:
            flag to apply the hierarchy to the discrete predictions to ensure consistency
    '''
    def __init__(self, name, threshold = .5, type2id = None, transitive = False):
        
        self.threshold = self.init_threshold(threshold)
        self.type2id = type2id
        self.type2id_inference = type2id
        self.num_class_inf = len(self.type2id)
        self.transitive = transitive
        # prepare map to fill missing predictions
        if self.transitive:
            if type2id:
                self.transitive_inference_mat = self.get_transitive_inference_mat(type2id)
            else:
                raise Exception('Error: you must provide type2id when transitive=True')
    
    def init_threshold(self, threshold):
        if str(threshold).lower() == 'auto':
            threshold = .5
            self.calibrate_threshold = True
        else:
            self.calibrate_threshold = False
        return threshold

    def infer_types(self, network_output):
        discrete_pred = self.discretize_output(network_output)
        # complete the predictions according to the hierarchy
        if self.transitive:
            discrete_pred = self.apply_hierarchy(discrete_pred)
        return discrete_pred

    def discretize_output(self, network_output):
        mask = network_output > self.threshold
        ones = torch.ones(mask.shape, device=network_output.device.type)
        zeros = torch.zeros(mask.shape, device=network_output.device.type)
        discrete_pred = torch.where(mask, ones, zeros)
        return discrete_pred.type(torch.int8)
    
    def apply_hierarchy(self, discrete_pred):
        if discrete_pred.device.type == 'cuda' and self.transitive_inference_mat.device.type != 'cuda':
            self.transitive_inference_mat = self.transitive_inference_mat.cuda()
        # apply transitivity to the ancestors in the hierarchy
        discrete_pred = torch.matmul(discrete_pred.to(torch.float32), self.transitive_inference_mat)
        # clip discret preds > 1 (deriving from consistent predictions)
        discrete_pred = discrete_pred.clip(0, 1)
        return discrete_pred.type(torch.int8)
    
    def get_transitive_inference_mat(self, type2id):
        # create ancestors map
        ancestors_map = {}
        for t in type2id.keys():
            splitted_path = t.replace('/',' /').split(' ')[1:]
            ancestors_map[t] = []
            prev_ancestor = ''
            for ancestor in splitted_path[:-1]:
                ancestor = prev_ancestor + ancestor
                ancestors_map[t].append(ancestor)    
                prev_ancestor = ancestor
        
        # transform ancestors map into a matrix
        n_types = len(type2id)
        transitive_inference_mat = torch.zeros((n_types, n_types))
        for t, ancestors in ancestors_map.items():
            idx_t = type2id[t]
            transitive_inference_mat[idx_t, idx_t] = 1
            for ancestor in ancestors:
                idx_ancestor =  type2id[ancestor]
                transitive_inference_mat[idx_t,idx_ancestor] = 1
        
        return transitive_inference_mat 

    def transform_true_types(self, true_types):
        return true_types

class MaxInferenceManager(BaseInferenceManager):
    def __init__(self, **kwargs):
        super().__init__(threshold=0, **kwargs)

    def discretize_output(self, network_output):
        ids = torch.argmax(network_output, dim = 1)
        num_classes = network_output.shape[1]
        discrete_pred = F.one_hot(ids, num_classes)
        return discrete_pred.type(torch.int8)

class MaxLeafInferenceManager(MaxInferenceManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.leaves_ids = self.get_leaves(self.type2id)
    
    def get_leaves(self, type2id):
        leaves_ids = []
        for t1 in type2id:
            is_leaf = True
            for t2 in type2id:
                if t1 != t2 and t1 in t2:
                    is_leaf = False
                    break
            if is_leaf:
                leaves_ids.append(type2id[t1])
        return leaves_ids
    
    def discretize_output(self, network_output):
        network_output = network_output[:, self.leaves_ids]
        return super().discretize_output(network_output)

class ThresholdOrMaxInferenceManager(BaseInferenceManager):
    '''
    Inference manager to discretize the output of the network accordingly to a :code:`threshold`; subclass of :code:`BaseInferenceManager`.
    In addition, it avoids void predictions by assigning 1 to the max output of each row tensor in case of all zeros produced by the base inference.

    Parameters:
        threshold:
            threshold to use to discretize the output
    '''
    def discretize_output(self, network_output):
        # threshold inference
        discrete_pred_threshold = super().discretize_output(network_output)
        # max inference to avoid void predictions
        ids = torch.argmax(network_output, dim = 1)
        num_classes = network_output.shape[1]
        discrete_pred_max = F.one_hot(ids, num_classes)
        # aggregate
        discrete_pred = discrete_pred_threshold + discrete_pred_max
        discrete_pred = torch.where(discrete_pred >= 1, 1, 0)
        return discrete_pred.type(torch.int8)

class FlatToHierarchyThresholdOrMaxInferenceManager(ThresholdOrMaxInferenceManager):
    def __init__(self, name, threshold = .5, type2id = None):
        if not type2id:
            raise Exception('Error: you must provide type2id to perform inference on a flat dataset')

        self.threshold = self.init_threshold(threshold)
        self.type2id_flat = type2id
        self.type2id_original = get_type2id_original(type2id)
        # using a flat dataset requires transitivity to ensure consistent predictions
        self.transitive = True
        # prepare map to fill missing predictions
        self.transitive_inference_mat = self.get_transitive_inference_mat(self.type2id_original)

    def infer_types(self, network_output):
        # discretize flat predictions
        discrete_pred_flat = self.discretize_output(network_output)
        # convert flat predictions to match the original dataset
        discrete_pred_original = torch.zeros((discrete_pred_flat.shape[0], len(self.type2id_original)), device=discrete_pred_flat.device.type)
        for t_flat, idx_flat in self.type2id_flat.items():
            # if the type is /*/NIL convert it to father type and assign value
            if t_flat.endswith('/NIL'):
                t = t_flat[:-4]
            else: # the type is shared between flat dataset and original dataset
                t = t_flat
            # copy prediction
            idx = self.type2id_original[t]
            discrete_pred_original[:, idx] = discrete_pred_flat[:, idx_flat]
        # complete the predictions according to the hierarchy
        discrete_pred_original = self.apply_hierarchy(discrete_pred_original)
        return discrete_pred_original

    def transform_true_types(self, true_types):
        return self.infer_types(true_types)


class IncrementalThresholdOrMaxInferenceManager(ThresholdOrMaxInferenceManager):
    def discretize_output(self, network_output):
        network_output_pretraining, network_output_incremental = network_output
        # apply ThresholdOrMaxInferenceManager on the predictions of the pretrained classifier
        discrete_pred_pretraining = super().discretize_output(network_output_pretraining)
        # incremental inference: apply ThresholdOrMaxInferenceManager only for predictions that were empty
        discrete_pred_pretraining_base = BaseInferenceManager.discretize_output(self, network_output_pretraining)
        discrete_pred_incremental = BaseInferenceManager.discretize_output(self, network_output_incremental)
        for i in (torch.sum(discrete_pred_pretraining_base, dim=1) == 0).nonzero():
          if torch.sum(discrete_pred_incremental[i]) == 0:
            discrete_pred_incremental[i] = super().discretize_output(network_output_incremental[i])
        # concatenate discretized predictions
        discrete_pred = torch.concat((discrete_pred_pretraining, discrete_pred_incremental), dim=1)
        return discrete_pred.type(torch.int8)

class IncrementalDoubleThresholdOrMaxInferenceManager(ThresholdOrMaxInferenceManager):
    def __init__(self, name, threshold_pretraining=0.5, threshold_incremental=0.5, type2id=None, transitive=False):
        super().__init__(name, None, type2id, transitive)
        self.threshold_pretraining = self.init_threshold(threshold_pretraining)
        self.threshold_incremental = self.init_threshold(threshold_incremental)

    def discretize_output(self, network_output):
        network_output_pretraining, network_output_incremental = network_output
        
        # SELECT threshold_pretraining
        self.threshold = self.threshold_pretraining
        # apply ThresholdOrMaxInferenceManager on the predictions of the pretrained classifier
        discrete_pred_pretraining = super().discretize_output(network_output_pretraining)

        # SELECT threshold_incremental
        self.threshold = self.threshold_incremental
        # incremental inference: apply ThresholdOrMaxInferenceManager only for predictions that were empty
        discrete_pred_pretraining_base = BaseInferenceManager.discretize_output(self, network_output_pretraining)
        discrete_pred_incremental = BaseInferenceManager.discretize_output(self, network_output_incremental)
        for i in (torch.sum(discrete_pred_pretraining_base, dim=1) == 0).nonzero():
          if torch.sum(discrete_pred_incremental[i]) == 0:
            discrete_pred_incremental[i] = super().discretize_output(network_output_incremental[i])
        # concatenate discretized predictions
        discrete_pred = torch.concat((discrete_pred_pretraining, discrete_pred_incremental), dim=1)
        return discrete_pred.type(torch.int8)
    
class ALIGNIEInferenceManager(BaseInferenceManager):
    def __init__(self, **kwargs):
        super().__init__(threshold=0, **kwargs)

        son_father_dict = defaultdict(list)
        father_labels = []
        for flat_label in self.type2id.keys():
            sub_label = flat_label.split('/')[1:]
            father = '/'
            for i in range(len(sub_label)-1):
                father += sub_label[i] + '/'
                son_father_dict[flat_label].append(father[:-1])
                father_labels.append(father[:-1])
        
        extra_label = sorted(list(set(father_labels) - set(self.type2id.keys())))
        all_labels = list(self.type2id.keys()) + extra_label
        
        self.type2id_inference = {label:i for i, label in enumerate(all_labels)}

        self.inference_idx = {}
        for k, v in son_father_dict.items():
            self.inference_idx[self.type2id_inference[k]] = [self.type2id_inference[k]] + [self.type2id_inference[father] for father in v]
            
        self.num_class_inf = len(self.type2id_inference)

    def discretize_output(self, network_output):
        flat_ids = torch.argmax(network_output, dim = 1)
        full_ids = flat_ids.new([self.inference_idx[idx.item()] for idx in flat_ids])
        discrete_pred = F.one_hot(full_ids, len(self.type2id_inference)).sum(dim=1)
        return discrete_pred.type(torch.int8)
    
    def transform_true_types(self, true_types):
        flat_ids = torch.argmax(true_types, dim = 1)
        full_ids = flat_ids.new([self.inference_idx[idx.item()] for idx in flat_ids])
        transformed_types = F.one_hot(full_ids, len(self.type2id_inference)).sum(dim=1)
        return transformed_types

