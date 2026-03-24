"""Download required MediaPipe .task model files.

Idempotent: skips files that already exist and verifies file size after download.
Run with: python setup_models.py  OR  make setup
"""

import os
import sys
import urllib.request
import urllib.error

# Map local filename → remote URL.
# Filenames must match what tracker.py expects.
MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

# Minimum expected size in bytes (models are several MB; reject truncated downloads)
_MIN_MODEL_SIZE = 500_000


def download_models(target_dir: str | None = None) -> bool:
    """Download MediaPipe .task files if they don't already exist.

    Args:
        target_dir: Directory to save models in. Defaults to repo root
                    (same directory as this script).

    Returns:
        True if all models are present after running, False on any failure.
    """
    if target_dir is None:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    print("--- IrisFlow Model Setup ---")
    all_ok = True

    for filename, url in MODELS.items():
        filepath = os.path.join(target_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > _MIN_MODEL_SIZE:
            print(f"  [OK] {filename} already exists ({os.path.getsize(filepath):,} bytes)")
            continue

        print(f"  Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  [FAIL] {filename}: {exc}")
            all_ok = False
            continue

        size = os.path.getsize(filepath)
        if size < _MIN_MODEL_SIZE:
            print(f"  [WARN] {filename} looks truncated ({size:,} bytes)")
            all_ok = False
        else:
            print(f"  [OK] {filename} downloaded ({size:,} bytes)")

    return all_ok


if __name__ == "__main__":
    success = download_models()
    if success:
        print("\nSetup complete. Run with: python -m src")
    else:
        print("\nSome models failed to download. Check your network and retry.")
        sys.exit(1)
