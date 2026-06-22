# download_model.py
import os
import gdown
from pathlib import Path

def download_model():
    """Download model dari Google Drive"""
    
    # Ganti dengan ID file Google Drive Anda
    # Cara dapatkan ID: 
    # 1. Upload model ke Google Drive
    # 2. Dapatkan link sharing
    # 3. Ambil ID dari link: https://drive.google.com/file/d/[ID]/view
    
    FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"  # GANTI INI!
    OUTPUT_DIR = Path("model_axis_indobart")
    OUTPUT_FILE = OUTPUT_DIR / "pytorch_model.bin"
    
    # Buat folder jika belum ada
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Download jika file belum ada
    if not OUTPUT_FILE.exists():
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, str(OUTPUT_FILE), quiet=False)
        print("✅ Model downloaded!")
    else:
        print("✅ Model already exists!")

if __name__ == "__main__":
    download_model()