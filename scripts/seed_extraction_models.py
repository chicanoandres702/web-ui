import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import ensure_default_extraction_models

if __name__ == "__main__":
    print("Seeding default extraction models...")
    ensure_default_extraction_models()
    print("Done. Models located in ./tmp/extraction_models")