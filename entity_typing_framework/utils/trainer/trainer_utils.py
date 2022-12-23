import torch
from tqdm import tqdm
import numpy as np

def calibrate_threshold(trainer, step=.025, metric='dev/macro_example/f1', patience=1000, incremental=False):
  if trainer.model.inference_manager.calibrate_threshold:
      # compute patience as 10% of the total number of steps
      if patience == 'auto':
        patience = int(1000 // (step*1000) * .1)

      counter = 0

      # disable validation metrics flag
      trainer.model.is_calibrating_threshold = True
      # produce network output for the dev set
      trainer.model.trainer.validate(trainer.model.trainer.model, trainer.model.trainer.datamodule.val_dataloader())
      network_output_for_inference = trainer.model.dev_network_output
      true_types = trainer.model.dev_true_types
      # iterate over thresholds and call validation routine
      max_metric = 0
      max_t = 0
      for t in tqdm(np.arange(step, 1, step), desc='Calibrating the threshold'):
          print()
          print('Validating model with threshold set to', t)
          if incremental:
            trainer.model.inference_manager.threshold_incremental = t
            inferred_types = trainer.model.inference_manager.infer_types(network_output_for_inference)
            idx = torch.tensor(list(trainer.model.type2id_incremental.values()))
            inferred_types_filtered = inferred_types.index_select(dim=1, index=idx.cuda())
            trainer.model.incremental_only_metric_manager.update(inferred_types_filtered.cuda(), true_types.cuda())
            metrics = trainer.model.incremental_only_metric_manager.compute()
          else:
            trainer.model.inference_manager.threshold = t
            inferred_types = trainer.model.inference_manager.infer_types(network_output_for_inference)
            trainer.model.metric_manager.update(inferred_types.cuda(), true_types.cuda())
            metrics = trainer.model.metric_manager.compute()
          current_metric = metrics[metric]
          print()
          print(f'{metric}:', current_metric.item())
          if current_metric >= max_metric:
              print('Value improved')
              max_metric = current_metric
              max_t = t
              counter = 0
          else:
              print('Value not improved')
              counter += 1

      print('*'*20)
      print(f'Optimal threshold {max_t}')
      print('*'*20)
      # set optimal threshold
      if incremental:
        trainer.model.inference_manager.threshold_incremental = max_t
      else:
        trainer.model.inference_manager.threshold = max_t