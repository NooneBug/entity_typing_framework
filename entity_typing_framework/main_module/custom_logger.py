from pytorch_lightning.callbacks.base import Callback
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import yaml



class CustomLogger(WandbLogger, Callback):
    def __init__(self, project, entity, name, config_file):
        # config = self.read_yaml_as_config(config_file)
        # wandb.init(project=project, entity=entity, name = name, config = config)
        wandb.init(project=project, entity=entity, name = name)
        super().__init__(project=project, entity=entity, name=name)
        self.to_log = {}
    
    def read_yaml_as_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml

    def log_all_metrics(self, metrics):
        for key, value in metrics.items():
            self.add(key=key, value=value)

    def log_loss(self, name, value):
        self.add(key=name, value=value)

    def add(self, key, value):
        self.to_log[key] = value

    def log_all(self):
        wandb.log(self.to_log)
        self.to_log = {}