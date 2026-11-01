from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


EXPECTED_CSV_SHA256 = "f737e3a0f2ce3d60fe43ece2cb458c795ec9940700a3cc9dd56f86df8f441540"
EXPECTED_ARCHIVE_SHA256 = "d261023a6bfb1335ee9dc40bba3401ec864576bc926df116343021f62d793017"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "psam_p53.csv.zip"
PARTS = sorted((ROOT / "data" / "raw").glob("psam_p53.csv.zip.part*"))
CSV = ROOT / "data" / "raw" / "psam_p53.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not CSV.is_file() or sha256(CSV) != EXPECTED_CSV_SHA256:
        rebuild_archive()
        extract_csv()

    actual_hash = sha256(CSV)
    if actual_hash != EXPECTED_CSV_SHA256:
        raise SystemExit(
            f"Dataset hash mismatch. Expected {EXPECTED_CSV_SHA256}, got {actual_hash}."
        )

    print(f"Dataset ready: {CSV}")
    print(f"SHA-256: {actual_hash}")


def rebuild_archive() -> None:
    if len(PARTS) != 3:
        raise SystemExit(
            "Dataset upload is incomplete. Expected three files named "
            "psam_p53.csv.zip.part01 through part03."
        )

    for part in PARTS:
        if part.read_bytes()[:32].startswith(b"version https://git-lfs"):
            raise SystemExit(
                f"{part.name} is a Git LFS pointer. Run `git lfs pull` first."
            )

    with ARCHIVE.open("wb") as combined:
        for part in PARTS:
            with part.open("rb") as source:
                shutil.copyfileobj(source, combined, length=1024 * 1024)

    archive_hash = sha256(ARCHIVE)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        ARCHIVE.unlink(missing_ok=True)
        raise SystemExit(
            "Dataset archive is incomplete or corrupted. "
            f"Expected {EXPECTED_ARCHIVE_SHA256}, got {archive_hash}."
        )


def extract_csv() -> None:
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extract("psam_p53.csv", CSV.parent)
    except zipfile.BadZipFile as error:
        raise SystemExit(
            "The reconstructed dataset archive is not a valid ZIP file. "
            "Re-upload all three archive parts."
        ) from error


if __name__ == "__main__":
    main()

