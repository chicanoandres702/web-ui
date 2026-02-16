import sys
import os
from installer.dependencies import DependencyManager
from installer.environment import SystemConfigurator
from installer.security import SecurityConfigurator
from installer.launcher import ApplicationLauncher

class Installer:
    """Main orchestrator."""
    def __init__(self):
        self.python = sys.executable
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.req_file = os.path.join(self.root, "requirements.txt")
        
        self.deps = DependencyManager(self.python, self.req_file)
        self.sys_config = SystemConfigurator(self.python)
        self.security = SecurityConfigurator(os.path.join(self.root, "certs"))
        self.launcher = ApplicationLauncher(self.python, self.root)

    def run(self):
        print("\n--- Scholar Agent Pro Installer ---")
        self.deps.upgrade_pip()
        self.deps.install_packages()
        
        self.sys_config.setup_windows_optimization()
        self.sys_config.setup_playwright()
        
        self.security.setup_certificates()
        
        self.launcher.launch()

if __name__ == "__main__":
    Installer().run()
