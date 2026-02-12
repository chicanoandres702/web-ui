"""
Main installation script for the project.
"""
from installation.installer import create_files, install_dependencies, run_application

def main():
    """
    Runs the complete installation and setup process.
    """
    create_files()
    install_dependencies()
    run_application()

if __name__ == "__main__":
    main()
