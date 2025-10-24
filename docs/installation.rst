Installation
============

Requirements
------------

* Python 3.11 or higher
* PyTorch 2.0 or higher
* CUDA-capable GPU (recommended for best performance)

Install from PyPI
-----------------

.. code-block:: bash

   pip install finsimtorch

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/simu-ai/finsimtorch.git
   cd finsimtorch
   pip install -e .

Install with Poetry
-------------------

.. code-block:: bash

   git clone https://github.com/simu-ai/finsimtorch.git
   cd finsimtorch
   poetry install

Development Installation
------------------------

For development, install with all dependencies:

.. code-block:: bash

   git clone https://github.com/simu-ai/finsimtorch.git
   cd finsimtorch
   poetry install --with dev
   pre-commit install
