

from entity_typing_framework.dataset_classes.datasets import BaseDataset


class DatasetManager():
    def __init__(self, dataset_paths) -> None:
        self.dataset_paths = dataset_paths

        self.dataset_routine()

    def dataset_routine(self):
        self.dataset = BaseDataset(dataset_paths = self.dataset_paths)