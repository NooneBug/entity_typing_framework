import torch

class BaseInferenceManager():
    '''
    Inference manager to discretize the output of the network accordingly to a :code:`threshold`.
    If an output value is greater than :code:`threshold`, then assign 1; assign 0 otherwise.

    Parameters:
        threshold:
            threshold to use to discretize the output
    '''
    def __init__(self, name, threshold = .5):
        self.threshold = threshold

    def infer_types(self, network_output):
        discrete_pred = self.discretize_output(network_output)
        return discrete_pred

    def discretize_output(self, network_output):
        mask = (network_output > self.threshold).cuda()
        ones = torch.ones(mask.shape).cuda()
        zeros = torch.zeros(mask.shape).cuda()
        discrete_pred = torch.where(mask, ones, zeros)
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