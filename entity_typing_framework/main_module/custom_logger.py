from pytorch_lightning.callbacks.base import Callback
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import yaml
import os



class CustomLogger(WandbLogger, Callback):
    def __init__(self, project, entity, name, config_file):
        config = self.read_yaml_as_config(config_file) if config_file != 'auto' else {}
        wandb.init(project=project, entity=entity, name=name, config=config)
        super().__init__(project=project, entity=entity, name=name)
        self.to_log = {}
    
    def log_config(self, config_dir, config_name='config.yaml'):
        with open(os.path.join(config_dir, config_name), 'r') as stream:
            try:
                config=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        wandb.config.update(config)

    def read_yaml_as_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml

    def on_fit_start(self, trainer, pl_module):
        config_dir = trainer.model.logger.experiment.get_logdir()
        self.log_config(config_dir)
        return super().on_fit_start(trainer, pl_module)

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
        
    def log_dataframe(self, df, prefix='test'):
        log_df = wandb.Table(data=df)
        wandb.log({f'{prefix}/final_metrics/table':log_df})