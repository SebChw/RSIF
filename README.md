Isolation Similarity Forest
==============================

We use [poetry](https://python-poetry.org/) as our dependency management tool so you need to install it first:
```sh
$ curl -sSL https://install.python-poetry.org | python3 -
```

Then clone this repo and go inside:
```sh
$ git clone https://github.com/SebChw/Isolation-Similarity-Forest.git
$ cd Isolation-Similarity-Forest
```

Create virtual environment and install all dependencies:
```sh
$ poetry shell
$ poetry install
```

Install `RISF` in editable mode (Otherwise you would need to have a path to it in your PYTHON_PATH to import it)
```sh
poetry run pip install -e .
```

To run all tests with coverage report
```sh
poetry run pytest --cov-report html --cov=risf tests
```

To run just unit tests
```sh
    poetry run pytest -m "not integration"
```

To run just integration tests. Please if you add new integration test add `@pytest.mark.integration` integration to it
```sh
    poetry run pytest -m "integration"
```

Important aspects during development:
* as you create new branch pleas do it in a format `[your_name]-[what you implement]`
* IF YOU RUN TESTS DO IT WITH `poetry run python -m pytest` this is important not to have problems with imports
* distinghuish between `poetry add` and `poetry add -D`, the latter is when a package is used only for development 
* `Commit your poetry.lock file to version control!` -> Then everyone has exactly the same dependencies

# Testing
To test, you should also install `tox`.

```sh 
$ pip install tox 
```
Then you can just run it, it'll trigger all tests.
```sh
$ tox
```

# How to get PyGod running after everything else is configured:
```sh
$ poetry shell
```
We need pytorch 1.12:
```sh
$ poetry run pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```
```sh
$ poetry run pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```
A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------