# data/download_thor.py
from __future__ import annotations

import os
import stat
import zipfile
from pathlib import Path

import requests

URL = (
    "http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/"
    "thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip"
)

HERE = Path(__file__).resolve().parent
ZIP_PATH = HERE / Path(URL).name


def main():
    # 1. download zip (if not exists)
    if not ZIP_PATH.exists():
        print(f"Downloading {URL}")
        r = requests.get(URL)
        r.raise_for_status()
        ZIP_PATH.write_bytes(r.content)
        print(f"Saved to {ZIP_PATH}")
    else:
        print(f"Zip already exists: {ZIP_PATH}")

    # 2. extract zip to current directory
    print("Extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(HERE)

    # 3. find .x86_64 executable and chmod +x
    exe = None
    for p in HERE.rglob("*.x86_64"):
        exe = p
        break

    if exe is None:
        raise RuntimeError("No .x86_64 executable found after extraction")

    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # 4. print result
    print("\n✅ Done")
    print("Unity executable path:")
    print(exe.resolve())
    print("\nUse it like:")
    print(f'  Ai2ThorRepresentation(executable_path="{exe.resolve()}")')


if __name__ == "__main__":
    main()
