import sys
import subprocess

def install_dependencies():
    print("\\n--- Installing Dependencies ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_application():
    print("\\n--- Launching Application ---")
    try:
        subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt:
        print("\\nStopped.")

if __name__ == "__main__":
    install_dependencies()
    run_application()
