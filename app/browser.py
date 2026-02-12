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

async def start_chrome_with_debug_port(port: int = 9222, headless: bool = False):
    """Start a browser with remote debugging enabled."""
    
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

    browser_exe = None
    for path in browser_paths:
        try:
            # Use asyncio.to_thread with subprocess.run to avoid blocking and NotImplementedError
            result = await asyncio.to_thread(
                subprocess.run,
                [path, '--version'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                browser_exe = path
                break
        except (FileNotFoundError, OSError):
            continue
    
    if not browser_exe:
        raise RuntimeError('Could not find a compatible browser (Edge, Chrome, or Chromium). Please install one and ensure it is in your system`s PATH.')

    # usecr_data_dir = tempfile.mkdtemp(prefix="gemini_browser_")
    # usecr_data_dir = tempfile.tempdir("batman_browser")
    cmd = [
        browser_exe,
        #f'--user-data-dir={user_data_dir}',
        '--profile-directory=BatmanBrowser',
        f'--remote-debugging-port={port}',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-extensions',
        '--disable-popup-blocking',
    ]
    
    if headless:
        cmd.append('--headless=new')
    else:
        cmd.append('about:blank')

    logger.info(f"Launching Browser: {' '.join(cmd)}")
    # Use asyncio.to_thread with subprocess.Popen to avoid blocking and NotImplementedError
    process = await asyncio.to_thread(
        subprocess.Popen,
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    cdp_ready = False
    for i in range(20):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{port}/json/version', timeout=2) as response:
                    if response.status == 200:
                        cdp_ready = True
                        break
        except Exception:
            await asyncio.sleep(1)
        if i % 5 == 0: logger.info("Waiting for CDP...")

    if not cdp_ready:
        try: 
            process.terminate()
        except: pass
        # shutil.rmtree(user_data_dir, ignore_errors=True)
        raise RuntimeError('Browser failed to start with CDP.')
        
    logger.info("CDP Ready.")
    return process, user_data_dir

class BrowserAgentWrapper:
    def __init__(self, llm):
        self.llm = llm
        self.browser = None
        self.session = None
        self.chrome_process = None
        self.user_data_dir = None
    
    def set_provider(self, provider: str):
        """Sets the LLM provider for the agent wrapper."""
        self.llm.provider = provider
        logger.info(f"Provider set to: {self.provider}")

    async def start_session(self, headless: bool = False):
        if not HAS_LIBS: 
            logger.error(f"Cannot start session. Missing libs: {IMPORT_ERROR}")
            return

        logger.info(f"Initializing Session (Headless: {headless})")
        
        try:
            self.chrome_process, self.user_data_dir = await start_chrome_with_debug_port(9222, headless)
        except Exception as e:
            logger.error(f"Failed to launch browser process: {e}")
            logger.error(traceback.format_exc())
            raise

        try:
            self.browser = Browser(cdp_url="http://localhost:9222", headless=headless)
            try:
                # Primary attempt: Use the modern method (Playwright standard)
                self.session = await self.browser.new_context()
            except AttributeError:
                try:
                    # Fallback 1: Use the legacy method if new_context is missing
                    self.session = await self.browser.new_session()
                except AttributeError:
                    # Fallback 2: The browser object itself IS the session/context
                    self.session = self.browser
            if os.path.exists(COOKIE_PATH):
                try:
                    with open(COOKIE_PATH, 'r') as f:
                        cookies = json.load(f)
                    await self.session.add_cookies(cookies)
                    logger.info("Cookies loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load cookies: {e}")

        except Exception as e:
            logger.error(f"Failed to connect to browser: {e}")
            await self.close_session()
            raise

    async def close_session(self):
        if self.session:
            try: await self.session.close()
            except: pass
            self.session = None
        if self.browser:
            try: await self.browser.close()
            except: pass
            self.browser = None
        if self.chrome_process:
            logger.info("Terminating browser process...")
            try:
                self.chrome_process.terminate()
                # wait() is blocking, run it in a thread
                await asyncio.to_thread(self.chrome_process.wait)
            except Exception as e:
                logger.warning(f"Error killing process: {e}")
            self.chrome_process = None
        
        if self.user_data_dir:
            shutil.rmtree(self.user_data_dir, ignore_errors=True)
            logger.info(f"Cleaned up user data directory: {self.user_data_dir}")
            self.user_data_dir = None

    async def decompose_task(self, task: str, class_name: str):
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
