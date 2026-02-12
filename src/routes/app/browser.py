import json
import logging
import asyncio
import base64
import os
import sys
import subprocess
import tempfile

# Force Windows Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from typing import List, Optional, Any

# Libraries
try:
    import aiohttp
    from browser_use import Agent as BUAgent
    from browser_use.browser.browser import Browser, BrowserConfig
    from browser_use.browser.context import BrowserContextConfig
    from langchain.schema import HumanMessage
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    class BrowserConfig: 
        def __init__(self, **kwargs): pass
    class BrowserContextConfig: 
        def __init__(self, **kwargs): pass

from app.models import SubTask 

logger = logging.getLogger(__name__)

COOKIE_PATH = "./tmp/cookies.json"

async def start_chrome_with_debug_port(port: int = 9222, headless: bool = False):
    """
    Start Chrome with remote debugging enabled (Template Logic).
    Returns the Chrome process.
    """
    user_data_dir = tempfile.mkdtemp(prefix='chrome_cdp_')
    
    # Platform-specific paths
    chrome_paths = [
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', # macOS
        '/usr/bin/google-chrome', # Linux
        'chrome', # Windows PATH
        'chromium', 
        # Common Windows Paths
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
    ]
    
    chrome_exe = None
    for path in chrome_paths:
        if os.path.exists(path) or path in ['chrome', 'chromium']:
            try:
                # Test executable
                test_proc = await asyncio.create_subprocess_exec(
                    path, '--version', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                await test_proc.wait()
                chrome_exe = path
                break
            except Exception:
                continue
                
    if not chrome_exe:
        logger.error("Chrome not found in standard paths.")
        # Fallback to just 'chrome' and hope it's in PATH
        chrome_exe = 'chrome'

    cmd = [
        chrome_exe,
        f'--remote-debugging-port={port}',
        f'--user-data-dir={user_data_dir}',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-extensions',
        '--disable-popup-blocking',
        '--disable-blink-features=AutomationControlled'
    ]
    
    if headless:
        cmd.append('--headless=new')
    else:
        cmd.append('about:blank')

    logger.info(f"Launching Chrome: {cmd}")
    process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for CDP to be ready
    cdp_ready = False
    for i in range(20): # 20 second timeout
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{port}/json/version', timeout=1) as response:
                    if response.status == 200:
                        cdp_ready = True
                        break
        except Exception:
            pass
        if i % 2 == 0: logger.info("Waiting for CDP...")
        await asyncio.sleep(1)

    if not cdp_ready:
        try:
            process.terminate()
        except: pass
        raise RuntimeError('Chrome failed to start with CDP')
        
    logger.info("Chrome CDP Ready.")
    return process

class BrowserAgentWrapper:
    def __init__(self, llm):
        self.llm = llm
        self.browser = None
        self.context = None
        self.chrome_process = None

    async def start_session(self, headless: bool = False):
        if not HAS_LIBS: return

        logger.info(f"Initializing Session (Headless: {headless})")
        
        # 1. Launch Chrome Process Manually
        try:
            self.chrome_process = await start_chrome_with_debug_port(9222, headless)
        except Exception as e:
            logger.error(f"Failed to launch chrome process: {e}")
            # Fallback to letting browser-use try to spawn it naturally if manual fails
            pass

        # 2. Connect to the running CDP instance
        # Note: If self.chrome_process failed, cdp_url might fail, 
        # so we'll only set cdp_url if we launched it or want to try default localhost
        cdp_url = "http://localhost:9222"
        
        try:
            # Initialize Browser-Use with the CDP URL
            browser_config = BrowserConfig(
                cdp_url=cdp_url,
                disable_security=True
            )
            self.browser = Browser(config=browser_config)

            # 3. Create Context
            # We skip specific window size args here because the chrome process 
            # is already running, but we can set viewport in context
            context_config = BrowserContextConfig(
                browser_window_size={'width': 1280, 'height': 1100}
            )
            
            try:
                self.context = await self.browser.new_context(config=context_config)
            except TypeError:
                self.context = await self.browser.new_context()
                
            # 4. Load Cookies
            if os.path.exists(COOKIE_PATH):
                try:
                    with open(COOKIE_PATH, 'r') as f:
                        cookies = json.load(f)
                    await self.context.context.add_cookies(cookies)
                    logger.info("Cookies loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load cookies: {e}")

        except Exception as e:
            logger.error(f"Failed to connect to browser: {e}")
            if self.chrome_process:
                self.chrome_process.terminate()
            raise e

    async def close_session(self):
        """Closes the persistent session and kills the process."""
        if self.context:
            await self.context.close()
            self.context = None
            
        if self.browser:
            await self.browser.close()
            self.browser = None
            
        if self.chrome_process:
            logger.info("Terminating Chrome process...")
            try:
                self.chrome_process.terminate()
                await self.chrome_process.wait()
            except Exception as e:
                logger.warning(f"Error killing chrome: {e}")
            self.chrome_process = None

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
            if callback: await callback("log", "Browser libs missing. Skipping step.")
            return True

        # Ensure session is started
        if not self.browser or not self.context:
            await self.start_session(headless)

        try:
            agent = BUAgent(
                task=f"Context: {class_name}. Instruction: {description}", 
                llm=self.llm, 
                browser_context=self.context
            )
            
            run_task = asyncio.create_task(agent.run(max_steps=10))
            
            while not run_task.done():
                try:
                    page = await self.context.get_current_page()
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
