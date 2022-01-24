Introduction
============

This package supports the definition of a modular network to perform entity typing.

The idea behind a modular network is to not have a monolithic code for each project related to entity typing, but to uniform the code using the same package and add modules and submodules to implement specific architectures and/or behaviors

This package is built above `PyTorchLightning <https://www.pytorchlightning.ai/>`_ and `LightningCLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_, be sure to be confident with these frameworks before using entity_typing_framework

Network Configuration
---------------------

Since the network has to be modular, a key component of this package is the :doc:`configuration file <configuration_file>`

The configuration file is used to set the parameters of each module in the network; some default configuration files are stored in the `/default_configurations` folder on `github <https://github.com/NooneBug/entity_typing_framework.git>`_

Expected Modules
-----------------
The modules expected to appear in each entity typing project are the following:

:ref:`DatasetManager <DatasetManager>`
    A module that read a dataset and tokenize it, according to the expected format of data of the other modules (in particular the Encoder, the Type Encoder and the Loss).
    
    The dataset is expected to be divided in partitions (commonly train, validation, and test).

    Following `LightningCLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ the DatasetManager has to be a subclass of `Pytorch Lightning LightningDataModule <https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html>`_.
    
    This module is expected to produce a :code:`torch.utils.data.Dataset` for each dataset partition, and return a `torch.utils.dataloader.data.Dataloader` foe each dataset partition

:ref:`Encoder <Encoder>
    A module that encodes the input sentences formatted by the DatasetManager. 
    
    This module is expected to produce a :code:`torch.Tensor`