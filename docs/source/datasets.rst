Datasets
========

List of implemented :ref:`dataset <dataset>` classes, submodule of :ref:`DatasetManager <DatasetManager>`

The Datasets submodules have to be a subclass of :code:`torch.utils.data.Dataset` and have to prepare the :ref:`tokenized dataset <dataset_tokenizer>` to be used in a :code:`torch.utils.data.Dataloader`

Each submodule defines the :code:`batch` composition through the method :code:`__getitem__`


.. _ET_Dataset:

.. autoclass:: entity_typing_framework.dataset_classes.datasets_for_dataloader.ET_Dataset
    :members:
    :special-members: __getitem__, __len__