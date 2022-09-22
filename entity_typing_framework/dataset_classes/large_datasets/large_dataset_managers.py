from entity_typing_framework.dataset_classes.dataset_managers import DatasetManager, ELMoDatasetManager
import os

# TODO: 1) implement ELMo specializtion (multiple inheritance?)
# or
# TODO: 2) delete class and use the yaml to specify slice_dirpath, 
# and move get_tokenizer_config_name into tokenized_dataset
class DatasetManagerLarge(DatasetManager):

  def __init__(self, dataset_paths: dict, dataset_reader_params: dict, tokenizer_params: dict, dataset_params: dict, dataloader_params: dict, rw_options: dict):
    super().__init__(dataset_paths, dataset_reader_params, tokenizer_params, dataset_params, dataloader_params, rw_options)
    slice_dirpath = os.path.join(self.rw_options['dirpath'], self.tokenizer_config_name)
    self.tokenizer_params['slice_dirpath'] = slice_dirpath
    
class ELMoDatasetManagerLarge(ELMoDatasetManager, DatasetManagerLarge):
  def get_tokenizer_config_name(self):
        config_name = f"elmo_T{self.tokenizer_params['max_tokens']}"
        if 'slice_dimension' in self.tokenizer_params:
            config_name += f"_SD{self.tokenizer_params['slice_dimension']}"
        config_name += '_light' if self.rw_options['light'] else ''

        return config_name