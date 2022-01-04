import torch

class InferenceManager():
    def __init__(self, threshold = .5):
        self.threshold = threshold

    def infer_types(self, network_output):
        mask = network_output > self.threshold

        ones = torch.ones(mask.shape).cuda()
        zeros = torch.zeros(mask.shape).cuda()

        discrete_pred = torch.where(mask, ones, zeros)

        max_values_and_indices = torch.max(network_output, dim = 1)

        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        
        return discrete_pred