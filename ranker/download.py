"""Download the Amazon Shopping Queries (ESCI) dataset into data/."""

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/"
FILES = [
    "shopping_queries_dataset_examples.parquet",
    "shopping_queries_dataset_products.parquet",
]
DATA = Path("data")


def main():
    DATA.mkdir(exist_ok=True)
    for name in FILES:
        path = DATA / name
        if path.exists():
            print(f"have {path}")
            continue
        print(f"downloading {name}")
        tmp = path.with_suffix(".part")
        urlretrieve(BASE + name, tmp)
        tmp.rename(path)


if __name__ == "__main__":
    main()
