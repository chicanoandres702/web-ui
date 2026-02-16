import json
import logging
import asyncio
import base64
import os
import sys
import subprocess
import tempfile
import traceback
from typing import List, Optional, Any
import shutil

# Force Windows Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- IMPORT CHECK BLOCK ---
try:
    import aiohttp
    from browser_use import Agent as BUAgent, Browser
    from langchain_core.messages import HumanMessage
    HAS_LIBS = True
    IMPORT_ERROR = None
except ImportError as e:
    HAS_LIBS = False
    IMPORT_ERROR = f"{e}/n{traceback.format_exc()}"
    class Browser: pass

from app.models import SubTask 

COOKIE_PATH = "./tmp/cookies.json"

class BrowserFinder:
    """Locates a compatible browser executable."""
    @staticmethod
    async def find() -> Optional[str]:
        browser_paths = [
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/usr/bin/google-chrome',
        'msedge',
        'microsoft-edge',
        'chrome',
        'chromium',
    ]

        for path in browser_paths:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [path, '--version'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, OSError):
                continue
        return None

class ChromeLauncher:
    """Manages the Chrome/Edge process lifecycle."""
    def __init__(self, port: int = 9222, headless: bool = False, user_data_dir: Optional[str] = None):
        self.port = port
        self.headless = headless
        self.process = None
        self.user_data_dir = user_data_dir
        self.is_temp_profile = user_data_dir is None

    async def _is_debugger_running(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{self.port}/json/version', timeout=0.5) as response:
                    return response.status == 200
        except Exception:
            return False

    async def launch(self):
        if await self._is_debugger_running():
            logger.info(f"Chrome debugger already active on port {self.port}. Attaching to existing session.")
            return None, self.user_data_dir

        browser_exe = await BrowserFinder.find()
        if not browser_exe:
            raise RuntimeError('Could not find a compatible browser.')

        if self.is_temp_profile:
            self.user_data_dir = tempfile.mkdtemp(prefix="browser_use_")
        else:
            if self.user_data_dir:
                os.makedirs(self.user_data_dir, exist_ok=True)
        
        cmd = [
            browser_exe,
            f'--user-data-dir={self.user_data_dir}',
            f'--remote-debugging-port={self.port}',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-extensions',
            '--disable-popup-blocking',
            '--disable-blink-features=AutomationControlled',
        ]
        
        if self.headless:
            cmd.append('--headless=new')
        else:
            cmd.append('about:blank')

        logger.info(f"Launching Browser: {' '.join(cmd)}")
        
        self.process = await asyncio.to_thread(
            subprocess.Popen,
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if await self._wait_for_cdp():
            logger.info("CDP Ready.")
            return self.process, self.user_data_dir
        
        self.cleanup()
        raise RuntimeError('Browser failed to start with CDP.')

    async def _wait_for_cdp(self) -> bool:
        for i in range(20):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{self.port}/json/version', timeout=2) as response:
                        if response.status == 200:
                            return True
            except Exception:
                await asyncio.sleep(1)
            if i % 5 == 0: logger.info("Waiting for CDP...")
        return False

    def cleanup(self):
        if self.process:
            try: self.process.terminate()
            except: pass
        
        if self.is_temp_profile and self.user_data_dir:
            shutil.rmtree(self.user_data_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp user data dir: {self.user_data_dir}")

class CookieService:
    """Manages cookie persistence."""
    def __init__(self, cookie_path: str = COOKIE_PATH):
        self.cookie_path = cookie_path

    async def load_cookies(self, session: Any):
        if os.path.exists(self.cookie_path):
            try:
                with open(self.cookie_path, 'r') as f:
                    cookies = json.load(f)
                if hasattr(session, 'add_cookies'):
                    await session.add_cookies(cookies)
                logger.info("Cookies loaded.")
            except Exception as e:
                logger.warning(f"Failed to load cookies: {e}")

class PlaywrightService:
    """Manages the Playwright/CDP connection."""
    def __init__(self, cdp_url: str, headless: bool):
        self.cdp_url = cdp_url
        self.headless = headless
        self.browser = None
        self.context = None

    async def connect(self) -> Any:
        self.browser = Browser(cdp_url=self.cdp_url, headless=self.headless)
        try:
            # Primary attempt: Use the modern method (Playwright standard)
            self.context = await self.browser.new_context()
        except AttributeError:
            try:
                # Fallback 1: Use the legacy method if new_context is missing
                self.context = await self.browser.new_session()
            except AttributeError:
                # Fallback 2: The browser object itself IS the session/context
                self.context = self.browser
        return self.context

    async def close(self):
        if self.context:
            try: await self.context.close()
            except: pass
        if self.browser:
            try: await self.browser.close()
            except: pass

class TaskDecomposer:
    """Handles LLM-based task decomposition."""
    def __init__(self, llm):
        self.llm = llm

    async def decompose(self, task: str, class_name: str):
        if not HAS_LIBS:
            return [SubTask(description=f"Analyze {task}")]

        prompt = f"Role: Research Assistant. Break academic task '{task}' for context '{class_name}' into 3 distinct, executable web-browsing steps. Return ONLY a raw JSON list of strings."
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0: 
                raise ValueError("No JSON list found in response")
            steps = json.loads(content[start:end])
            return [SubTask(description=s) for s in steps]
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return [SubTask(description=f"Research: {task}")]

class BrowserSessionManager:
    def __init__(self, headless: bool = False, persistent: bool = True):
        self.headless = headless
        self.persistent = persistent
        self.chrome_process = None
        self.user_data_dir = None
        self.browser = None
        self.session = None
        self.launcher = None
        self.playwright_service = None
        self.cookie_service = CookieService()

    async def start_session(self):
         logger.info(f"Initializing Session (Headless: {self.headless})")
        
         try:
            user_data_dir = None
            if self.persistent:
                user_data_dir = os.path.abspath(os.path.join(os.getcwd(), "chrome_data"))
                logger.info(f"Using persistent profile at: {user_data_dir}")

            self.launcher = ChromeLauncher(port=9222, headless=self.headless, user_data_dir=user_data_dir)
            self.chrome_process, self.user_data_dir = await self.launcher.launch()
         except Exception as e:
            logger.error(f"Failed to launch browser process: {e}")
            logger.error(traceback.format_exc())
            raise

         try:
            self.playwright_service = PlaywrightService(cdp_url="http://localhost:9222", headless=self.headless)
            self.session = await self.playwright_service.connect()
            self.browser = self.playwright_service.browser
            await self.cookie_service.load_cookies(self.session)
         except Exception as e:
            logger.error(f"Failed to connect to browser: {e}")
            await self.close_session()
            raise

    async def close_session(self):
        if self.playwright_service:
            await self.playwright_service.close()
        if self.launcher:
            self.launcher.cleanup()

class BrowserAgentWrapper:
    def __init__(self, llm):
        self.llm = llm
        self.decomposer = TaskDecomposer(llm)
        self.browser = None
        self.session = None
        self.chrome_process = None
        self.user_data_dir = None
        self.session_manager = None
    

    def set_provider(self, provider: str):
        """Sets the LLM provider for the agent wrapper."""
        self.llm.provider = provider
        logger.info(f"Provider set to: {self.provider}")

    async def start_session(self, headless: bool = False, persistent: bool = True):
        if not HAS_LIBS: 
            logger.error(f"Cannot start session. Missing libs: {IMPORT_ERROR}")
            return
        
        try: 
            self.session_manager = BrowserSessionManager(headless, persistent)
            await self.session_manager.start_session()
            self.browser = self.session_manager.browser
            self.session = self.session_manager.session
            self.chrome_process = self.session_manager.chrome_process
            self.user_data_dir = self.session_manager.user_data_dir
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            await self.close_session()
            raise e

    async def close_session(self):
        if self.session_manager:
            await self.session_manager.close_session()
        
        self.session = None
        self.browser = None
        self.chrome_process = None
        self.user_data_dir = None

    async def decompose_task(self, task: str, class_name: str):
        return await self.decomposer.decompose(task, class_name)

    async def run_step(self, description: str, class_name: str, headless: bool = False, quality: int = 50, callback=None) -> bool:
        if not HAS_LIBS:
            msg = f"Browser libs missing. Error: {IMPORT_ERROR}"
            if callback: await callback("log", msg)
            return False

        if not self.browser or not self.session:
            await self.start_session(headless)

        try:
            agent = BUAgent(
                task=f"Context: {class_name}. Instruction: {description}", 
                llm=self.llm, 
                browser_context=self.session
            )
            
            run_task = asyncio.create_task(agent.run(max_steps=10))
            
            while not run_task.done():
                try:
                    page = await self.session.get_current_page()
                    if page:
                        screenshot = await page.screenshot(type='jpeg', quality=quality)
                        encoded = base64.b64encode(screenshot).decode('utf-8')
                        if callback:
                            await callback("browser_stream", {
                                "image": encoded,
                                "url": page.url,
                                "title": await page.title()
                            })
                except Exception:
                    pass 
                await asyncio.sleep(0.5) 
            
            await run_task
            return True

        except Exception as e:
            logger.error(f"Browser Execution Error: {e}")
            if callback: await callback("log", f"Browser Error: {e}")





            return False
