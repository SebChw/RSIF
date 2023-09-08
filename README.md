Isolation Similarity Forest
==============================

# Reproduction steps: 
1. Download precomputed distances matrices from [Zenodo](https://zenodo.org/record/8328048). Unzip downloaded zip archive into the repository foldery.
2. Intall RSIF:
```sh
$ pip install -r requirements/requirements.txt
$ pip install -r requirements/requirements_experiments.txt
$ pip install .
```
3. Now you have 2 choices:
   1. To reproduce both selection of best distances and experiments run `python clean_for_reproduction.py --clean_best`. This will remove `results`, `best_distances` and `figures` folders. Selection of best distances is done with nested repeated holdout. So it takes some time.
   2. To reproduce just experiments run `python clean_for_reproduction.py`. This will remove `results` and `figures` folder.


Next run all cells in:
1. `notebooks/sensitivity_analysis.ipynb`
2. `notebooks/experiments.ipynb`
3. `notebooks/visualizations.ipynb`


# Development

Please get familiar with the documentation inside `docs/build/html/index.html`
For now jupyter with examples is not ready yet. Check `tests/test_integration.py` or `notebooks/utils` to see example usage of RISF
To use this module

Clone this repo and go inside:
```sh
$ git clone https://github.com/SebChw/Isolation-Similarity-Forest.git
$ cd Isolation-Similarity-Forest
```

Create virtual environment and install needed dependencies:

If you only want to use the library
```sh
$ pip install -r requirements/requirements.txt
```

If you want to develop the library additionally
```sh
$ pip install -r requirements/requirements_dev.txt
```

If you want to run experiments
```sh
$ pip install -r requirements/requirements_experiments.txt
```

If you want to contribute to documentation:
```sh
$ conda install sphinx
$ pip install -r requirements/requirements_docs.txt

```
 To build documentation
 ```sh
$ cd docs
$ make clean
$ make html
 ```

Install `RISF` in editable mode (Otherwise you would need to have a path to it in your PYTHON_PATH to import it)
```sh
pip install -e .
```

To run all tests with coverage report
```sh
pytest --cov-report html --cov=risf tests
```

To run just unit tests
```sh
pytest -m "not integration"
```

To run just integration tests. Please if you add new integration test add `@pytest.mark.integration` integration to it
```sh
pytest -m "integration"
```

# Testing
To test, you should also install `tox`.

```sh 
$ pip install tox 
```
Then you can just run it, it'll trigger all tests.
```sh
$ tox
```