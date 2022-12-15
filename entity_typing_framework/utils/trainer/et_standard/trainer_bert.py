from entity_typing_framework.dataset_classes.dataset_managers import DatasetManager, ELMoDatasetManager
from entity_typing_framework.main_module.custom_logger import CustomLogger
from entity_typing_framework.main_module.main_module import KENNMainModule, MainModule, KENNMultilossMainModule
from pytorch_lightning.utilities.cli import LightningCLI
# from entity_typing_framework.dataset_classes.large_datasets.large_dataset_managers import DatasetManagerLarge
import trainer_utils

def dummy_compute_fn(arg):
    return arg

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(CustomLogger, "logger")

        parser.link_arguments("data.type_number", "model.type_number", apply_on="instantiate")
        parser.link_arguments("data.type2id", "model.type2id", apply_on="instantiate")
        parser.link_arguments("logger", "model.logger", apply_on="instantiate", compute_fn = dummy_compute_fn)
        
cli = MyLightningCLI(MainModule, DatasetManager, save_config_overwrite=True)
# load best model
cli.trainer.model.ET_Network = cli.trainer.model.load_ET_Network(checkpoint_to_load=cli.trainer.checkpoint_callback.best_model_path, ET_Network_params = cli.trainer.model.ET_Network_params)

# calibrate threshold
trainer_utils.calibrate_threshold(cli.trainer, 0.025)

# test step
cli.trainer.test(cli.trainer.model, cli.trainer.datamodule.test_dataloader())
