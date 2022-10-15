# Isolation-Similarity-Forest

We use [poetry](https://python-poetry.org/) as our dependency management tool so you need to install it first:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Then clone this repo and go inside:
```sh
git clone https://github.com/SebChw/Isolation-Similarity-Forest.git
cd Isolation-Similarity-Forest
```

Create virtual environment and install all dependencies:
```sh
poetry shell
poetry install
```
Important aspects during development:
* as you create new branch pleas do it in a format `[your_name]-[what you implement]`
* IF YOU RUN TESTS DO IT WITH `poetry run python -m pytest` this is important not to have problems with imports
* distinghuish between `poetry add` and `poetry add -D`, the latter is when a package is used only for development 
* `Commit your poetry.lock file to version control!` -> Then everyone has exactly the same dependencies
