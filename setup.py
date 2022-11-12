# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
    ['risf', 'risf.utils']

package_data = \
    {'': ['*']}

install_requires = \
    ['black>=22.10.0,<23.0.0',
     'joblib>=1.2.0,<2.0.0',
     'pandas>=1.5.1,<2.0.0',
     'pytest-cov>=4.0.0,<5.0.0',
     'scikit-learn>=1.1.3,<2.0.0']

setup_kwargs = {
    'name': 'random-isolation-similarity-forest',
    'version': '0.1.0',
    'description': 'reaserch project developing new tool for outlier detection.',
    'long_description': "Isolation Similarity Forest\n==============================\n\nWe use [poetry](https://python-poetry.org/) as our dependency management tool so you need to install it first:\n```sh\n$ curl -sSL https://install.python-poetry.org | python3 -\n```\n\nThen clone this repo and go inside:\n```sh\n$ git clone https://github.com/SebChw/Isolation-Similarity-Forest.git\n$ cd Isolation-Similarity-Forest\n```\n\nCreate virtual environment and install all dependencies:\n```sh\n$ poetry shell\n$ poetry install\n```\n\nInstall `ISF` in editable mode (Otherwise you would need to have a path to it in your PYTHON_PATH to import it)\n```sh\npoetry run pip install -e .\n```\n\nTo run all tests with coverage report\n```sh\npoetry run pytest --cov-report html --cov=isf tests\n```\n\nImportant aspects during development:\n* as you create new branch pleas do it in a format `[your_name]-[what you implement]`\n* IF YOU RUN TESTS DO IT WITH `poetry run python -m pytest` this is important not to have problems with imports\n* distinghuish between `poetry add` and `poetry add -D`, the latter is when a package is used only for development \n* `Commit your poetry.lock file to version control!` -> Then everyone has exactly the same dependencies\n\n# Testing\nTo test, you should also install `tox`.\n\n```sh \n$ pip install tox \n```\nThen you can just run it, it'll trigger all tests.\n```sh\n$ tox\n```\n\n\nA short description of the project.\n\nProject Organization\n------------\n\n    ├── LICENSE\n    ├── Makefile           <- Makefile with commands like `make data` or `make train`\n    ├── README.md          <- The top-level README for developers using this project.\n    ├── data\n    │\xa0\xa0 ├── external       <- Data from third party sources.\n    │\xa0\xa0 ├── interim        <- Intermediate data that has been transformed.\n    │\xa0\xa0 ├── processed      <- The final, canonical data sets for modeling.\n    │\xa0\xa0 └── raw            <- The original, immutable data dump.\n    │\n    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details\n    │\n    ├── models             <- Trained and serialized models, model predictions, or model summaries\n    │\n    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),\n    │                         the creator's initials, and a short `-` delimited description, e.g.\n    │                         `1.0-jqp-initial-data-exploration`.\n    │\n    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.\n    │\n    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.\n    │\xa0\xa0 └── figures        <- Generated graphics and figures to be used in reporting\n    │\n    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.\n    │                         generated with `pip freeze > requirements.txt`\n    │\n    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported\n    ├── src                <- Source code for use in this project.\n    │\xa0\xa0 ├── __init__.py    <- Makes src a Python module\n    │   │\n    │\xa0\xa0 ├── data           <- Scripts to download or generate data\n    │\xa0\xa0 │\xa0\xa0 └── make_dataset.py\n    │   │\n    │\xa0\xa0 ├── features       <- Scripts to turn raw data into features for modeling\n    │\xa0\xa0 │\xa0\xa0 └── build_features.py\n    │   │\n    │\xa0\xa0 ├── models         <- Scripts to train models and then use trained models to make\n    │   │   │                 predictions\n    │\xa0\xa0 │\xa0\xa0 ├── predict_model.py\n    │\xa0\xa0 │\xa0\xa0 └── train_model.py\n    │   │\n    │\xa0\xa0 └── visualization  <- Scripts to create exploratory and results oriented visualizations\n    │\xa0\xa0     └── visualize.py\n    │\n    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io\n\n\n--------",
    'author': 'SebChw',
    'author_email': 'sebastian.chwilczynski@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
