import json


class BaseDataset:
    '''
    This class manages an entire dataset, commonly partitioned in train, dev and test.
    
    The initialization expects all partitions (e.g., train, dev and test) paths organized in a dictionary
    
    Parameters
        dataset_paths : 
            a dictionary with keys the partition names (e.g., train, dev, test) and with values the paths to the partitions

            The values are taken from the :code:`yaml` config file following the key :code:`data.dataset_paths` 

    the :code:`__init__` saves the parameters and automatic calls :code:`read_dataset_partitions()`
    '''
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths

        self.read_dataset_partitions()
    
    def read_dataset_partitions(self):
        '''
        Instances a :code:`DatasetPartition` for each partition found in :code:`dataset_paths`
        '''
        partitions = {}

        for partition_name, partition_path in self.dataset_paths.items():
            partitions[partition_name] = DatasetPartition(partition_path=partition_path)

        self.partitions = partitions

class DatasetPartition:
    '''
        Manages a dataset partition, saved in a json file 

        Parameters
            partition_path:
                a path in which is stored a json dataset
        
        The :code:`__init__` method calls :code:`acquire_data()`
    '''
    def __init__(self, partition_path):
        self.path = partition_path
        self.acquire_data()
    
    def acquire_data(self):
        '''
            Acquires data from a dataset stored in a txt file in which each line is a json with keys:

            mention_span:
                a string defining the entity mention

            left_context_token:
                a list of strings definining the left context of the entity mention

            right_context_token:
                a list of strings definining the right context of the entity mention
             
            y_str:
                a list of strings defining the types which the entity mention in this context belongs to
        '''
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
        '''
        Returns the number of elements found in the dataset file
        '''
        return len(self.mentions)
