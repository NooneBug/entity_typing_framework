import torch

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
        self.threshold = threshold
        self.type2id = type2id
        self.transitive = transitive
        # prepare map to fill missing predictions
        if self.transitive:
            self.ancestors_map = self.get_ancestors_map(type2id)

    def infer_types(self, network_output):
        discrete_pred = self.discretize_output(network_output)
        # complete the predictions according to the hierarchy
        if self.transitive:
            discrete_pred = self.apply_hierarchy(discrete_pred)
        return discrete_pred

    def discretize_output(self, network_output):
        mask = (network_output > self.threshold).cuda()
        ones = torch.ones(mask.shape).cuda()
        zeros = torch.zeros(mask.shape).cuda()
        discrete_pred = torch.where(mask, ones, zeros)
        return discrete_pred

    def apply_hierarchy(self, discrete_pred):
        # apply transitivity to the ancestors in the hierarchy
        for dp in discrete_pred:
            for t, ancestors in self.ancestors_map.items():
                idx_t = self.type2id[t]
                for ancestor in ancestors:
                    idx_ancestor = self.type2id[ancestor]
                    if dp[idx_t] == 1 and dp[idx_ancestor] == 0:
                        dp[idx_ancestor] = 1
        return discrete_pred
    
    def get_ancestors_map(self, type2id):
        ancestors_map = {}
        for t in type2id.keys():
            splitted_path = t.replace('/',' /').split(' ')[1:]
            ancestors_map[t] = []
            prev_ancestor = ''
            for ancestor in splitted_path[:-1]:
                ancestor = prev_ancestor + ancestor
                ancestors_map[t].append(ancestor)    
                prev_ancestor = ancestor
        return ancestors_map 

class MaxInferenceManager(BaseInferenceManager):
    def __init__(self, **kwargs):
        super().__init__(threshold=None, **kwargs)

    def discretize_output(self, network_output):
        max_values_and_indices = torch.max(network_output, dim = 1)
        discrete_pred = torch.zeros_like(network_output, device=network_output.device)
        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        return discrete_pred

class ThresholdOrMaxInferenceManager(BaseInferenceManager):
    '''
    Inference manager to discretize the output of the network accordingly to a :code:`threshold`; subclass of :code:`BaseInferenceManager`.
    In addition, it avoids void predictions by assigning 1 to the max output of each row tensor in case of all zeros produced by the base inference.

    Parameters:
        threshold:
            threshold to use to discretize the output
    '''
    def discretize_output(self, network_output):
        discrete_pred = super().discretize_output(network_output)
        # perform max inference to avoid void predictions
        max_values_and_indices = torch.max(network_output, dim = 1)
        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        return discrete_pred

class IncrementalThresholdOrMaxInferenceManager(ThresholdOrMaxInferenceManager):
    def discretize_output(self, network_output_pretraining, network_output_incremental):
        # apply ThresholdOrMaxInferenceManager on the predictions of the pretrained classifier
        discrete_pred_pretraining = super().discretize_output(network_output_pretraining)
        # incremental inference: apply ThresholdOrMaxInferenceManager only for predictions that were empty
        discrete_pred_pretraining_base = super(ThresholdOrMaxInferenceManager, self).discretize_output(network_output_pretraining)
        discrete_pred_incremental = super(ThresholdOrMaxInferenceManager, self).discretize_output(network_output_incremental)
        for i in (torch.sum(discrete_pred_pretraining_base, dim=1) == 0).nonzero():
          if torch.sum(discrete_pred_incremental[i]) == 0:
            discrete_pred_incremental[i] = super().discretize_output(network_output_incremental[i])
        # concatenate discretized predictions
        discrete_pred = torch.concat((discrete_pred_pretraining, discrete_pred_incremental), dim=1)
        return discrete_pred