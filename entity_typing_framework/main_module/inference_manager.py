import torch

class BaseInferenceManager():
    '''
    Inference manager to discretize the output of the network accordingly to a :code:`threshold`.
    If an output value is greater than :code:`threshold`, then assign 1; assign 0 otherwise.

    Parameters:
        threshold:
            threshold to use to discretize the output
    '''
    def __init__(self, name, threshold = .5, type2id = None):
        self.threshold = threshold
        self.type2id = type2id

    def infer_types(self, network_output):
        discrete_pred = self.discretize_output(network_output)
        return discrete_pred

    def discretize_output(self, network_output):
        mask = (network_output > self.threshold).cuda()
        ones = torch.ones(mask.shape).cuda()
        zeros = torch.zeros(mask.shape).cuda()
        discrete_pred = torch.where(mask, ones, zeros)
        return discrete_pred

class MaxInferenceManager(BaseInferenceManager):
    def __init__(self, name, type2id = None):
        self.type2id = type2id

    def discretize_output(self, network_output):
        max_values_and_indices = torch.max(network_output, dim = 1)
        discrete_pred = torch.zeros_like(network_output, device=network_output.device)
        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        return discrete_pred

class MaxTransitiveInferenceManager(MaxInferenceManager):
    
    def __init__(self, name, type2id=None):
        super().__init__(name, type2id)
        # create ancestors map utils
        ancestors_map = {}
        for t in type2id.keys():
            splitted_path = t.replace('/',' /').split(' ')[1:]
            ancestors_map[t] = []
            prev_ancestor = ''
            for ancestor in splitted_path[:-1]:
                ancestor = prev_ancestor + ancestor
                ancestors_map[t].append(ancestor)    
                prev_ancestor = ancestor
        self.ancestors_map = ancestors_map 

    def infer_types(self, network_output):
        discrete_pred = super().infer_types(network_output)
        # applying transitivity to the ancestors in the hierarchy
        for dp in discrete_pred:
            for t, ancestors in self.ancestors_map.items():
                idx_t = self.type2id[t]
                for ancestor in ancestors:
                    idx_ancestor = self.type2id[ancestor]
                    if dp[idx_t] == 1 and dp[idx_ancestor] == 0:
                        dp[idx_ancestor] = 1
                    

        return discrete_pred
    


class ThresholdOrMaxInferenceManager(BaseInferenceManager):
    '''
    Inference manager to discretize the output of the network accordingly to a :code:`threshold`; subclass of :code:`BaseInferenceManager`.
    In addition, it avoids void predictions by assigning 1 to the max output of each row tensor in case of all zeros produced by the base inference.

    Parameters:
        threshold:
            threshold to use to discretize the output
    '''
    def infer_types(self, network_output):
        discrete_pred = super().infer_types(network_output)
        # perform max inference to avoid void predictions
        max_values_and_indices = torch.max(network_output, dim = 1)
        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        return discrete_pred

class IncrementalThresholdOrMaxInferenceManager(ThresholdOrMaxInferenceManager):
    def infer_types(self, network_output_pretraining, network_output_incremental):
        # apply ThresholdOrMaxInferenceManager on the predictions of the pretrained classifier
        discrete_pred_pretraining = super().infer_types(network_output_pretraining)
        # incremental inference: apply ThresholdOrMaxInferenceManager only for predictions that were empty
        discrete_pred_pretraining_base = super(ThresholdOrMaxInferenceManager, self).infer_types(network_output_pretraining)
        discrete_pred_incremental = super(ThresholdOrMaxInferenceManager, self).infer_types(network_output_incremental)
        for i in (torch.sum(discrete_pred_pretraining_base, dim=1) == 0).nonzero():
          if torch.sum(discrete_pred_incremental[i]) == 0:
            discrete_pred_incremental[i] = super().infer_types(network_output_incremental[i])
        # concatenate discretized predictions
        discrete_pred = torch.concat((discrete_pred_pretraining, discrete_pred_incremental), dim=1)
        return discrete_pred