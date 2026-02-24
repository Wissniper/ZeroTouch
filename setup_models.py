import os
import urllib.request

MODELS = {
    "face_landmarker_v2_with_blendshapes.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
}

def download_models():
    """Downloads the required MediaPipe .task files if they don't exist."""
    print("--- IrisFlow Model Setup ---")
    
    for filename, url in MODELS.items():
        if os.path.exists(filename):
            print(f"{filename} already exists.")
        else:
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_models()
    print("\nSetup complete. You can now run: python src/main.py")
