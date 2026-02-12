"""
This module contains the installation logic for the project.
"""
import os
import sys
import subprocess
from installation.files import FILES

def create_files():
    """
    Creates the project files and directories.
    """
    print("/n--- 📂 Creating Project Files ---")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for filepath, content in FILES.items():
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Created: {filepath}")

def install_dependencies():
    """
    Installs the project dependencies.
    """
    print("/n--- 📦 Installing Dependencies ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("/n--- 🌐 Initializing Browser Components ---")
        subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        print("✅ Browser components initialized successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during dependency installation: {e}")
        sys.exit(1)

def run_application():
    """
    Runs the main application.
    """
    print("/n--- 🚀 Launching Application ---")
    try:
        subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt:
        print("/n🛑 Application stopped by user.")
