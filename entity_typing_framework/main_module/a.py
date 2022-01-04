import torchmetrics
import torch

pred = torch.tensor([[1., 0, 1],
                     [0, 0, 1]])

target = torch.tensor([[1, 0, 0],
                        [0, 0, 1]])

macro_p_ex = torchmetrics.Precision(num_classes=3, average='samples', mdmc_average='global')
macro_r_ex = torchmetrics.Precision(num_classes=3, average='samples', mdmc_average='global')
macro_f1_ex = torchmetrics.Precision(num_classes=3, average='samples', mdmc_average='global')


macro_p_ex.update(pred, target)
a = macro_p_ex.compute()
print(a)