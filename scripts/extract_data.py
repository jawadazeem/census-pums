from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


EXPECTED_SHA256 = "f737e3a0f2ce3d60fe43ece2cb458c795ec9940700a3cc9dd56f86df8f441540"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "psam_p53.csv.zip"
CSV = ROOT / "data" / "raw" / "psam_p53.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not ARCHIVE.is_file():
        raise SystemExit(f"Missing dataset archive: {ARCHIVE}")

    if not CSV.is_file() or sha256(CSV) != EXPECTED_SHA256:
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extract("psam_p53.csv", CSV.parent)

    actual_hash = sha256(CSV)
    if actual_hash != EXPECTED_SHA256:
        raise SystemExit(
            f"Dataset hash mismatch. Expected {EXPECTED_SHA256}, got {actual_hash}."
        )

    print(f"Dataset ready: {CSV}")
    print(f"SHA-256: {actual_hash}")


if __name__ == "__main__":
    main()

