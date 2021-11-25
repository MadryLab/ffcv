Making an FFCV dataloader
=========================

After :ref:`writing an FFCV dataset <Writing a dataset to FFCV format>`, we are
ready to start loading data (and training models)! We'll continue using the same
regression dataset as the previous guide, and we'll assume that the dataset has
been written to ``/path/to/data.beton``.

In order to load the dataset that we've written, we'll need the
:class:`ffcv.loader.Loader` class (which will do most of the heavy lifting), and
a set of *decoders* corresponding to the fields present in the dataset (so in
our case, we will use the :class:`~ffcv.fields.decoders.FloatDecoder` and
:class:`~ffcv.fields.decoders.NDArrayDecoder` classes):

.. code-block:: python

    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import SimpleRGBImageDecoder, FloatDecoder

Our first step is instantiating the :class:`~ffcv.loader.Loader` class:

.. code-block:: python

  loader = Loader('/path/to/dataset.beton',
                  batch_size=BATCH_SIZE,
                  num_workers=NUM_WORKERS,
                  order=ORDERING,
                  pipelines=PIPELINES)

In order to create a loader, we need to specify a path to the FFCV dataset,
batch size, number of workers, as well as two less standard arguments, ``order``
and ``pipelines``, which we discuss below:

Dataset ordering
''''''''''''''''
The ``order`` option in the loader initialization is similar to PyTorch
DataLoaders' ``shuffle`` option, with some additional optionality. This argument
takes an ``enum`` provided by :class:`ffcv.loader.OrderOption`:

.. code-block:: python

  from ffcv.loader import OrderOption

  # Truly random shuffling (shuffle=True in PyTorch)
  ORDERING = OrderOption.RANDOM

  # Unshuffled (i.e., served in the order the dataset was written)
  ORDERING = OrderOption.SEQUENTIAL

  # Memory-efficient but not truly random loading
  # <TODO: explain what it does>
  # Speeds up loading over RANDOM when the whole dataset does not fit in RAM!
  ORDERING = OrderOption.QUASIRANDOM

Pipelines
'''''''''
The ``pipeline`` option in :class:`ffcv.loader.Loader` specifies the dataset and
tells the loader what fields to read, how to read them, and what operations to
apply on top. Specifically, a pipeline is a key-value dictionary where the key
matches the one used in `writing the dataset <writing>`_, and the value is a
sequence of operations to perform. The operations must start with a
:class:`ffcv.fields.decoders.Decoder` object corresponding to that field. 
For example, the following pipeline reads the fields and then converts each one
to a PyTorch tensor:

.. code-block:: python 
  
  from ffcv.transforms import ToTensor

  PIPELINES = {
    'covariate': [NDArrayDecoder(), ToTensor()],
    'label': [FloatDecoder(), ToTensor()]
  }

This is already enough to start loading data, but pipelines are also our
opportunity to apply fast pre-processing to the data through a series of
transformations---transforms are automatically compiled to ---- at runtime 
and, for GPU-intensive applications like training neural networks, can often 
introduce no additional training overhead. 

.. note::
  
  In fact, declaring field pipelines is optional: for any field that the
  in the dataset file without a corresponding pipeline specified in the
  ``pipelines`` dictionary,  the :class:`~ffcv.loader.Loader` will default to
  the barebones pipeline above, i.e., first a decoder
  then a conversion to PyTorch tensor. (You can force FFCV to explicitly *not*
  load a field by adding a corresponding ``None`` entry to the ``pipelines``
  dictionary.) 

  If the entire ``pipelines`` argument is
  unspecified, this barebones pipeline will be applied to all fields. 
 
Transforms
"""""""""""

There are three easy ways to specify transformations in a pipeline:

- By default, FFCV implements a set of standard transformations in the
  :mod:`ffcv.transforms` module

- Any subclass of ``torch.nn.Module`` will be automatically converted to a FFCV
  operation

- You can easily implement your own transformations by subclassing
  :class:`ffcv.transforms.Operation`, as discussed in the `Making custom
  transforms <TODO>`_ guide.
