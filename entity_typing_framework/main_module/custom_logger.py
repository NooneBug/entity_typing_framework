from pytorch_lightning.callbacks.base import Callback
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

class CustomLogger(WandbLogger, Callback):
    def __init__(self, entity, **kwargs):
        project = kwargs['project']
        wandb.init(project=project, entity=entity)
        super().__init__(**kwargs)
        self.to_log = {}
    
    def log_all_metrics(self, metrics):
        for key, value in metrics.items():
            self.add(key, value)
        self.log_all()

    def add(self, key, value):
        self.to_log[key] = value

    def log_all(self):
        wandb.log(self.to_log)
        self.to_log = {}