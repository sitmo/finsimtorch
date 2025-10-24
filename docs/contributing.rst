Contributing
============

We welcome contributions to finsimtorch! Please read this guide before submitting any changes.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/finsimtorch.git
      cd finsimtorch

3. Install the development dependencies:

   .. code-block:: bash

      poetry install --with dev
      pre-commit install

4. Create a new branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Code Style
----------

We use several tools to maintain code quality:

* **Black** for code formatting
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking

Run these tools before committing:

.. code-block:: bash

   poetry run black .
   poetry run isort .
   poetry run flake8 finsimtorch tests
   poetry run mypy finsimtorch

Testing
-------

Write tests for any new functionality:

.. code-block:: bash

   poetry run pytest

Make sure all tests pass and maintain good test coverage.

Documentation
-------------

Update documentation for any new features:

1. Add docstrings to all public functions and classes
2. Update the API documentation in ``docs/api/``
3. Add examples in ``docs/examples/`` if applicable
4. Update the main documentation as needed

Pull Request Process
--------------------

1. Ensure all tests pass
2. Update documentation
3. Add a clear description of your changes
4. Reference any related issues
5. Request review from maintainers

Release Process
---------------

Releases are handled automatically via GitHub Actions when tags are pushed to the main branch.

For maintainers:
1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create and push a git tag
4. GitHub Actions will build and publish to PyPI
