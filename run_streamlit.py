# run_streamlit.py
import os
import subprocess
import sys

def main():
    print("=" * 50)
    print("  AXIS TRANSLATOR - STREAMLIT")
    print("=" * 50)
    print("Starting Streamlit server...")
    print("http://localhost:8501")
    print("=" * 50)
    
    # Jalankan streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    main()