from browser_use.browser.browser import Browser
from browser_use.browser.browser import BrowserConfig

class CustomBrowser(Browser):
    def __init__(self, config: BrowserConfig):
        super().__init__(config=config)