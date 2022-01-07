from torch.utils.data import Dataset
import torch
import json


class BaseDataset:
    '''
        this class manages an entire dataset, partitioned in train, dev and test.
    '''
    def __init__(self, dataset_paths):
        '''
        the initialization expects all partitions (e.g., train, dev and test) paths organized in a dictionary
            - dataset_paths : a dictionary with keys the partition names (e.g., train, dev, test) and with values the paths to the partitions
        '''
        self.dataset_paths = dataset_paths

        self.read_dataset_partitions()
    
    def read_dataset_partitions(self):
        
        partitions = {}

        for partition_name, partition_path in self.dataset_paths.items():
            partitions[partition_name] = DatasetPartition(partition_path=partition_path)

        self.partitions = partitions

class DatasetPartition:
    '''
        manages a dataset partition, saved in a json file 
    '''
    def __init__(self, partition_path):
        self.path = partition_path
        self.acquire_data()
    
    def acquire_data(self):
        mentions = []
        left_contexts = []
        right_contexts = []
        labels = []

        with open(self.path, 'r') as inp:
            lines = [json.loads(l) for l in inp.readlines()]

        for l in lines:
            mentions.append((l['mention_span']))
            left_contexts.append((" ".join(l['left_context_token'])))
            right_contexts.append((" ".join(l['right_context_token'])))
            labels.append(l['y_str'])

        self.mentions = mentions
        self.left_contexts = left_contexts
        self.right_contexts = right_contexts
        self.labels = labels
    
    def get_elements_number(self):

        return len(self.mentions)
