from pathlib import Path
import urllib.request
import zipfile


DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "ml-100k"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def generate_dataset():
    """
    Download and extract the MovieLens 100K dataset.
    """

    if DATASET_DIR.exists() and (DATASET_DIR / "u.data").exists():
        print("MovieLens 100K dataset already exists.")
        print(f"Dataset location: {DATASET_DIR.resolve()}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "ml-100k.zip"

    print("Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(ZIP_URL, zip_path)

    print("Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    zip_path.unlink(missing_ok=True)

    print("\nDataset generated successfully!")
    print(f"Dataset location: {DATASET_DIR.resolve()}")

    print("\nFiles:")
    for file in DATASET_DIR.iterdir():
        print(" -", file.name)


if __name__ == "__main__":
    generate_dataset()
