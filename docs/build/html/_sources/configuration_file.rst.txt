Configuration File
==================

A configuration file is a :code:`yaml` file defined following `LightningCLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ specifics. It is used to define each hyperparameter needed to drive the modular network definition, training and evaluation and the dataset management

Some example configuration files are stored in the `/default_configurations` folder on `github <https://github.com/NooneBug/entity_typing_framework.git>`_

Trainer Parameters
------------------
`LightningCLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ supports the parametrization of a `Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ object through a :code:`yaml` configuration file

The trainer parameters are defined under the :code:`yaml` key :code:`trainer`

For example:

.. code-block:: yaml

    trainer:
        gpus: 1
        limit_train_batches: .1
        max_epochs : 5
        ...

Default trainer parameters can be shown using :code:`python trainer.py fit --print-config`, see `LightningCLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_

General Parameters
------------------

Under `/default_configurations` files you cane notice that some parameters are given to `LightningCLI <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ without indentation. For this project, the useful LightningCLI parameters are:

seed_everything
  set the global seed in the python runtime (:code:`torch`, :code:`cuda` and :code:`numpy`)

Expected Modules and Parameters
-------------------------------

Each expected module has parameters; in the :code:`yaml` configuration file expected module parameters are organized under a :code:`key`

Here is reported the list expected modules together with each key:  

DatasetManager
    key: dataset_manager