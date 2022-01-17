Modules and submodules
======================

Here is reported the list of each implemented module and submodule the relative documentation

.. _DatasetManager:

DatasetManager
--------------

Implemented dataset managers are expected to manage the dataset acquisition, the tokenization and the formatting of a partitioned dataset. 

Commonly a dataset is partitioned in :code:`train`, :code:`validation` and :code:`test`

A dataset managers is expected to have the following submodules:

dataset_reader:
    a module that read the dataset from the actual dataset file

dataset_tokenizer:
    a module that tokenize the acquired dataset

dataset:
    a module that is a subclass of :code:`torch.utils.data.Dataset` and has to prepare the tokenized dataset to be used in a :code:`torch.utils.data.Dataloader`

Implemented Modules
-------------------

.. toctree::
    
    dataset_managers

Implemented Submodules
----------------------

.. toctree::
    
    dataset_readers
    dataset_tokenizers
    datasets

