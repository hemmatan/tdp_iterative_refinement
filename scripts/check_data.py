"""Check the exact corpus split bytes used by the paper configuration."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_data(data_dir):
    expected = json.loads((ROOT / "configs/data_hashes.json").read_text())
    errors = []
    for filename, record in expected.items():
        path = data_dir / filename
        if not path.is_file():
            errors.append(f"Missing {filename}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"Checksum differs: {filename}")
    if errors:
        raise ValueError("; ".join(errors))
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    try:
        check_data(args.data_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print("All three corpus splits match the paper source checksums.")


if __name__ == "__main__":
    main()
