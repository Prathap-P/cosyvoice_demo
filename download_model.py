"""
download_model.py
-----------------
One-time script to download the CosyVoice2-0.5B pretrained model from Hugging Face.
Run this AFTER installing requirements:
    python download_model.py

Model size: ~2 GB — this will take a few minutes depending on your connection.
The model will be saved to: pretrained_models/CosyVoice2-0.5B/
"""

import os
from huggingface_hub import snapshot_download

MODEL_REPO = "FunAudioLLM/CosyVoice2-0.5B"
LOCAL_DIR  = "pretrained_models/CosyVoice2-0.5B"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"Downloading {MODEL_REPO} → {LOCAL_DIR}")
    print("This may take several minutes (~2 GB)...\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=LOCAL_DIR,
    )

    print(f"\n✅ Model downloaded successfully to: {LOCAL_DIR}")
    print("You can now run:  python demo.py")

if __name__ == "__main__":
    main()
