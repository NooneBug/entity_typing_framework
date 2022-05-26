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
        mask = network_output > self.threshold
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

class BoxEmbeddingInferenceManager(ThresholdOrMaxInferenceManager):
    def infer_types(self, network_output):
        return super().infer_types(torch.exp(network_output))