import json
import logging
import asyncio
import base64
from app.features.agent.models import SubTask
from langchain.schema import HumanMessage

try:
    from browser_use import Agent as BUAgent
    from browser_use.browser.browser import Browser, BrowserConfig
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

logger = logging.getLogger(__name__)

class BrowserService:
    def __init__(self, llm):
        self.llm = llm

    async def decompose_task(self, task: str, class_name: str):
        prompt = f"Break academic task '{task}' for {class_name} into 3 steps. Return ONLY a JSON list of strings."
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)]).content
            start, end = res.find('['), res.rfind(']') + 1
            steps = json.loads(res[start:end])
            return [SubTask(description=s) for s in steps]
        except:
            return [SubTask(description=task)]

    async def run_step(self, description: str, class_name: str, headless: bool = False, quality: int = 50, callback=None) -> bool:
        if not HAS_LIBS: 
            logger.warning("Browser libraries not found.")
            await asyncio.sleep(1); return True
        
        browser = None
        ctx = None
        try:
            # Initialize Browser
            browser = Browser(config=BrowserConfig(headless=headless, disable_security=True))
            
            # Create Context
            ctx = await browser.new_context()
            agent = BUAgent(task=description, llm=self.llm, browser_context=ctx)
            
            run_task = asyncio.create_task(agent.run(max_steps=15))
            
            while not run_task.done():
                try:
                    page = await ctx.get_current_page()
                    if page:
                        screenshot = await page.screenshot(type='jpeg', quality=quality)
                        encoded = base64.b64encode(screenshot).decode('utf-8')
                        if callback:
                            await callback("browser_stream", {
                                "image": encoded,
                                "url": page.url,
                                "title": await page.title()
                            })
                except:
                    pass
                await asyncio.sleep(0.5)
            
            await run_task
            return True
        except Exception as e:
            logger.error(f"Browser Execution Error: {e}")
            return False
        finally:
            if ctx: await ctx.close()
            if browser: await browser.close()
