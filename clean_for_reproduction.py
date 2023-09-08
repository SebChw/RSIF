import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--clean_best",
    help="Beside results and figures also removes best_distances folder. Nested cross validation will be performed to find them.",
    action="store_true",
)


def clean_folder(folder: str):
    if not Path(folder).is_dir():
        raise ValueError(
            f"{folder} doesn't exist. Please run the script from the root of the project. Or if you already run this script run experiments first"
        )

    shutil.rmtree(folder)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.clean_best:
        print("removing best_distances folder")
        clean_folder("best_distances")
    else:
        print("keeping best_distances folder")

    print("removing results folder")
    clean_folder("results")
    print("removing figures folder")
    clean_folder("figures")
