Installation
============

.. note::
    We highly recommend to install parallelproj and pre-compiled versions of the libraries and the python interface from **conda-forge**.
    ``parallelproj`` depends on ``libparallelproj`` which is also available on conda-forge, in a cuda and non-cuda version.

.. tip::

   You can get the **miniforge** conda install (minimal conda installer specific to conda-forge) `here <https://github.com/conda-forge/miniforge>`_.
   As usual, we recommend to install parallelproj into a separate **virtual environment**.

To install parallelproj (and the required compiled libraries) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install parallelproj

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge parallelproj
