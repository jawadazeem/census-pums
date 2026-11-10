from pathlib import Path
import shutil
import zipfile


root = Path(__file__).resolve().parents[1]
raw_dir = root / "data" / "raw"
csv_path = raw_dir / "psam_p53.csv"
zip_path = raw_dir / "psam_p53.csv.zip"
parts = sorted(raw_dir.glob("psam_p53.csv.zip.part*"))


def main():
    if csv_path.exists():
        print(f"data is already ready: {csv_path}")
        return

    if len(parts) != 3:
        raise SystemExit("expected three dataset parts in data/raw")

    # put the split zip back together
    with zip_path.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)

    # unzip the original csv
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("psam_p53.csv", raw_dir)

    zip_path.unlink()
    print(f"data is ready: {csv_path}")


if __name__ == "__main__":
    main()
