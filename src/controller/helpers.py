import logging
import base64
import aiohttp
import os
from src.utils.browser_scripts import JS_WAIT_FOR_DOM_STABILITY, JS_CLOSE_COOKIE_BANNERS

logger = logging.getLogger(__name__)

async def smart_scan(page, script, args=None, check_empty=True):
    """
    Executes a script, and if results are empty, waits for DOM stability and retries.
    Used for extracting data from pages that might be hydrating.
    """
    try:
        # Execute first attempt
        result = await page.evaluate(script, args) if args else await page.evaluate(script)
    except Exception as e:
        logger.warning(f"Smart scan failed (likely page closed): {e}")
        return None
    
    should_retry = False
    if check_empty:
        if isinstance(result, list) and len(result) == 0:
            should_retry = True
        elif isinstance(result, dict) and not result:
            should_retry = True
            
    if should_retry:
        try:
            logger.info("Smart Scan: Empty result detected. Waiting for DOM stability...")
            await page.evaluate(JS_WAIT_FOR_DOM_STABILITY)
            result = await page.evaluate(script, args) if args else await page.evaluate(script)
        except Exception:
            pass
        
    return result

async def attempt_close_cookie_banner(page):
    """Helper to attempt closing cookie banners on main page and frames."""
    try:
        if await page.evaluate(JS_CLOSE_COOKIE_BANNERS): return True
    except Exception: pass
    
    for frame in page.frames:
        if frame == page.main_frame: continue
        try:
            if await frame.evaluate(JS_CLOSE_COOKIE_BANNERS): return True
        except Exception: pass
        
    return False

async def download_resource(page, url: str, filepath: str) -> str:
    """
    Downloads a resource using browser context first, then falling back to aiohttp.
    Returns success message or raises Exception.
    """
    # Strategy 1: Fetch via Page (Uses Browser Cookies/Session)
    try:
        js_script = """
            async (url) => {
                const response = await fetch(url);
                if (!response.ok) throw new Error(response.statusText);
                const blob = await response.blob();
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
            }
        """
        data_url = await page.evaluate(js_script, url)
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        
        data = base64.b64decode(encoded)
        with open(filepath, 'wb') as f:
            f.write(data)
        return f"Successfully downloaded {url} to {filepath}"
    except Exception as e_page:
        logger.warning(f"Page fetch failed for {url}: {e_page}. Trying fallback.")

    # Strategy 2: Fallback to aiohttp (Manual Cookies)
    try:
        cookies = await page.context.cookies()
        cookie_dict = {c['name']: c['value'] for c in cookies}
        
        async with aiohttp.ClientSession(cookies=cookie_dict) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    return f"Successfully downloaded {url} to {filepath}"
                else:
                    raise Exception(f"Status code: {response.status}")
    except Exception as e:
        raise Exception(f"All download strategies failed. Last error: {e}")