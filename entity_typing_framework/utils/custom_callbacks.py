from typing import Any, Dict, Optional, List
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


# class customModelCheckpoint(ModelCheckpoint):
#     def __init__(self, dirpath = None, filename: Optional[str] = None, monitor: Optional[str] = None, verbose: bool = False, save_last: Optional[bool] = None, save_top_k: int = 1, save_weights_only: bool = False, mode: str = "min", auto_insert_metric_name: bool = True, every_n_train_steps: Optional[int] = None, train_time_interval: Optional[timedelta] = None, every_n_epochs: Optional[int] = None, save_on_train_epoch_end: Optional[bool] = None, every_n_val_epochs: Optional[int] = None):
#         super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode, auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end, every_n_val_epochs)
    
#     def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> Dict[str, Any]:
#         return super().on_save_checkpoint(trainer, pl_module, checkpoint)

# class callback_to_predict(Callback):
#     def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]) -> None:
#         return super().on_predict_epoch_end(trainer, pl_module, outputs)