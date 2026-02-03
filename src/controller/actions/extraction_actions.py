import logging
import json
import os
from browser_use.browser.context import BrowserContext
from src.utils.utils import retry_async, save_text_to_file as utils_save_text
from src.utils.browser_scripts import (
    JS_EXTRACT_LINKS,
    JS_EXTRACT_TABLES,
    JS_ANALYZE_STRUCTURE,
    JS_EXTRACT_LIST_ITEMS,
    JS_GET_MAIN_CONTENT,
    JS_EXTRACT_IMAGES,
    JS_GET_ATTRIBUTES,
    JS_GET_ELEMENT_COORDINATES,
    JS_GET_PAGE_METADATA,
    JS_GET_LOCAL_STORAGE,
    JS_ENABLE_LOG_CAPTURE,
    JS_GET_CONSOLE_ERRORS,
    JS_GET_COMPUTED_STYLE,
    JS_FIND_TEXT_ELEMENTS,
    JS_GET_DROPDOWN_OPTIONS,
    JS_ASSESS_SECTION,
    JS_CHECK_TEXT_ELEMENT_STATUS
)
from src.controller.helpers import smart_scan
from src.agent.browser_use.smart_focus import focus_on_content

logger = logging.getLogger(__name__)

class ExtractionActionsMixin:
    def _register_extraction_actions(self):
        @self.registry.action("Extract all hyperlinks from the current page")
        async def extract_page_links(browser: BrowserContext):
            page = await browser.get_current_page()
            
            async def _extract():
                return await smart_scan(page, JS_EXTRACT_LINKS)

            try:
                links = await retry_async(
                    _extract,
                    logger=logger,
                    error_message="Link extraction failed"
                )
                
                # Deduplicate based on href
                unique_links = {}
                for link in links:
                    unique_links[link['href']] = link['text']
                
                formatted_links = [{"text": txt, "href": href} for href, txt in unique_links.items()]
                
                return f"Found {len(formatted_links)} unique links. Here are the top 50: {json.dumps(formatted_links[:50], indent=2)}"
            except Exception as e:
                return f"Failed to extract links after retries: {e}"

        @self.registry.action("Find navigation links relevant to a specific topic")
        async def find_navigation_options(browser: BrowserContext, topic: str):
            page = await browser.get_current_page()
            try:
                links = await smart_scan(page, JS_EXTRACT_LINKS)
                
                # Simple relevance scoring
                scored_links = []
                topic_lower = topic.lower()
                keywords = topic_lower.split()
                
                for link in links:
                    text = link['text'].lower()
                    href = link['href'].lower()
                    context = link.get('context', 'content')
                    score = 0
                    
                    # Semantic relevance
                    if topic_lower in text: score += 10
                    elif topic_lower in href: score += 5
                    
                    for kw in keywords:
                        if kw in text: score += 2
                        if kw in href: score += 1
                    
                    # Structural priority (Heuristic: Nav/Header links are usually better for general navigation)
                    if context in ['nav', 'header']: score += 3
                    if context == 'sidebar': score += 1
                    if context == 'footer': score -= 1 # De-prioritize footer unless specific
                        
                    if score > 0:
                        scored_links.append({**link, "score": score})
                
                scored_links.sort(key=lambda x: x['score'], reverse=True)
                
                if not scored_links:
                    return f"No links found relevant to '{topic}'."
                
                top_links = scored_links[:20]
                # Clean up output for LLM
                display_links = [{"text": l['text'], "href": l['href'], "context": l['context'], "score": l['score']} for l in top_links]
                return f"Found {len(scored_links)} relevant links. Top {len(top_links)}:\n{json.dumps(display_links, indent=2)}"
            except Exception as e:
                return f"Error finding navigation options: {e}"

        @self.registry.action("Extract and summarize tables from the page")
        async def extract_tables(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                tables = await smart_scan(page, JS_EXTRACT_TABLES)
                if not tables:
                    return "No visible tables found on this page."
                
                result = f"Found {len(tables)} tables:\n\n"
                for t in tables:
                    result += f"### Table {t['index']}: {t['caption'] or 'Untitled'}\n"
                    # Convert rows to Markdown
                    for i, row in enumerate(t['rows']):
                        result += "| " + " | ".join(row) + " |\n"
                        if i == 0: # Assume first row is header for formatting
                            result += "| " + " | ".join(["---"] * len(row)) + " |\n"
                    result += "\n"
                return result
            except Exception as e:
                return f"Error extracting tables: {e}"

        @self.registry.action("Analyze page structure (headings and interactive elements)")
        async def analyze_page_structure(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                structure = await smart_scan(page, JS_ANALYZE_STRUCTURE)
                # Format for readability
                summary = f"Headings: {len(structure.get('headings', []))}\nInteractive Elements: {len(structure.get('interactive', []))}\n\n"
                summary += json.dumps(structure, indent=2)
                return summary
            except Exception as e:
                return f"Error analyzing page structure: {e}"

        @self.registry.action("Extract list items from the page")
        async def extract_list_items(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                lists = await smart_scan(page, JS_EXTRACT_LIST_ITEMS)
                if not lists:
                    return "No significant lists found."
                return f"Found {len(lists)} lists:\n{json.dumps(lists, indent=2)}"
            except Exception as e:
                return f"Error extracting lists: {e}"

        @self.registry.action("Summarize the main content of the page")
        async def summarize_page(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                # Use heuristic to get main content, avoiding nav/footer noise
                content = await smart_scan(page, JS_GET_MAIN_CONTENT)
                return f"Main Content Summary ({len(content)} chars):\n{content[:3000]}..." 
            except Exception as e:
                return f"Error summarizing page: {e}"

        @self.registry.action("Extract visible images with their alt text and sources")
        async def extract_images(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                images = await smart_scan(page, JS_EXTRACT_IMAGES)
                if not images:
                    return "No significant images found."
                return f"Found {len(images)} images:\n{json.dumps(images, indent=2)}"
            except Exception as e:
                return f"Error extracting images: {e}"

        @self.registry.action("Extract text content from a specific container element")
        async def get_text_from_element(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                element = page.locator(selector).first
                if not await element.count():
                    return f"Element not found with selector: {selector}"
                
                text = await element.inner_text()
                return f"Text content from '{selector}':\n{text}"
            except Exception as e:
                return f"Error extracting text: {e}"

        @self.registry.action("Get specific attributes (e.g., href, src, class) of an element")
        async def get_element_attributes(browser: BrowserContext, selector: str, attributes: str = ""):
            page = await browser.get_current_page()
            try:
                data = await page.evaluate(JS_GET_ATTRIBUTES, [selector, attributes])
                if data is None:
                    return f"Element not found: {selector}"
                return f"Attributes for '{selector}':\n{json.dumps(data, indent=2)}"
            except Exception as e:
                return f"Error getting attributes: {e}"

        @self.registry.action("Get the bounding box coordinates of an element")
        async def get_element_coordinates(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                coords = await page.evaluate(JS_GET_ELEMENT_COORDINATES, selector)
                if not coords:
                    return f"Element not found: {selector}"
                return f"Coordinates for '{selector}':\n{json.dumps(coords, indent=2)}"
            except Exception as e:
                return f"Error getting coordinates: {e}"

        @self.registry.action("Get page metadata (SEO tags, OpenGraph)")
        async def get_page_metadata(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                data = await page.evaluate(JS_GET_PAGE_METADATA)
                return f"Page Metadata:\n{json.dumps(data, indent=2)}"
            except Exception as e:
                return f"Error getting metadata: {e}"

        @self.registry.action("Get the accessibility tree for the page")
        async def get_accessibility_tree(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                tree = await page.accessibility.snapshot(interesting_only=False)
                
                def filter_tree(node):
                    if not node: return None
                    
                    # Keep only relevant keys and filter out noise
                    filtered_node = {
                        "role": node.get("role"),
                        "name": node.get("name", "").strip(),
                    }
                    if not filtered_node["name"] and filtered_node["role"] in ['StaticText', 'generic', 'paragraph', 'text']:
                        return None
                    if filtered_node["role"] in ['none', 'presentation', 'img', 'graphics-document', 'document']:
                        return None

                    if "children" in node:
                        children = [filter_tree(child) for child in node["children"]]
                        filtered_node["children"] = [c for c in children if c]
                    
                    return filtered_node
                return f"Accessibility Tree (filtered):\n{json.dumps(filter_tree(tree), indent=2)}"
            except Exception as e:
                return f"Error getting accessibility tree: {e}"

        @self.registry.action("Get full HTML and attributes of a specific element")
        async def inspect_element_details(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                element = page.locator(selector).first
                if not await element.count():
                    return f"Element not found: {selector}"
                
                html = await element.evaluate("el => el.outerHTML")
                box = await element.bounding_box()
                return f"HTML:\n{html}\n\nBounding Box: {box}"
            except Exception as e:
                return f"Error inspecting element: {e}"

        @self.registry.action("Get all cookies from the current context")
        async def get_cookies(browser: BrowserContext):
            page = await browser.get_current_page()
            cookies = await page.context.cookies()
            return json.dumps(cookies, indent=2)

        @self.registry.action("Get cookies specifically for the current page URL")
        async def get_page_cookies(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                cookies = await page.context.cookies(urls=[page.url])
                return json.dumps(cookies, indent=2)
            except Exception as e:
                return f"Error getting page cookies: {e}"

        @self.registry.action("Set cookies for the current context")
        async def set_cookies(browser: BrowserContext, cookies_json: str):
            page = await browser.get_current_page()
            try:
                cookies = json.loads(cookies_json)
                if isinstance(cookies, dict):
                    cookies = [cookies]
                await page.context.add_cookies(cookies)
                return "Cookies set successfully."
            except Exception as e:
                return f"Error setting cookies: {e}"

        @self.registry.action("Get all local storage items from the current page")
        async def get_local_storage(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                data = await page.evaluate(JS_GET_LOCAL_STORAGE)
                return f"Local Storage:\n{json.dumps(data, indent=2)}"
            except Exception as e:
                return f"Error getting local storage: {e}"

        @self.registry.action("Enable console log capture on the current page")
        async def enable_console_log_capture(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.evaluate(JS_ENABLE_LOG_CAPTURE)
            return "Console log capture enabled. Use 'get_console_logs' to retrieve them."

        @self.registry.action("Get captured console logs")
        async def get_console_logs(browser: BrowserContext):
            page = await browser.get_current_page()
            logs = await page.evaluate("() => window._captured_logs || []")
            if not logs:
                return "No logs captured (did you run 'enable_console_log_capture'?)."
            return json.dumps(logs, indent=2)

        @self.registry.action("Get captured console errors only")
        async def get_console_errors(browser: BrowserContext):
            page = await browser.get_current_page()
            errors = await page.evaluate(JS_GET_CONSOLE_ERRORS)
            if not errors:
                return "No console errors captured."
            return json.dumps(errors, indent=2)

        @self.registry.action("Get computed style of an element to check visibility")
        async def get_element_computed_style(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                style = await page.evaluate(JS_GET_COMPUTED_STYLE, selector)
                
                if not style:
                    return f"Element not found: {selector}"
                
                return f"Computed Style for '{selector}':\n{json.dumps(style, indent=2)}"
            except Exception as e:
                return f"Error getting computed style: {e}"

        @self.registry.action("List all elements containing specific text")
        async def get_elements_by_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            try:
                elements = await smart_scan(page, JS_FIND_TEXT_ELEMENTS, args=text)
                return f"Found {len(elements)} visible elements containing '{text}':\n{json.dumps(elements, indent=2)}"
            except Exception as e:
                return f"Error finding elements by text: {e}"

        @self.registry.action("Get options from a dropdown/select element")
        async def get_dropdown_options(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                options = await smart_scan(page, JS_GET_DROPDOWN_OPTIONS, args=selector)
                
                if options is None:
                    return f"Element not found: {selector}"
                
                return f"Dropdown Options for '{selector}':\n{json.dumps(options, indent=2)}"
            except Exception as e:
                return f"Error getting options: {e}"

        @self.registry.action("Get HTML content of a specific iframe")
        async def get_iframe_content(browser: BrowserContext, frame_index: int):
            page = await browser.get_current_page()
            frames = page.frames
            if frame_index < 0 or frame_index >= len(frames):
                return f"Frame index {frame_index} out of bounds."
            try:
                content = await frames[frame_index].content()
                return f"Content of iframe {frame_index} (truncated):\n{content[:4000]}"
            except Exception as e:
                return f"Error getting content: {e}"

        @self.registry.action("Assess the current page section to determine next step (Scroll vs Submit)")
        async def assess_page_section(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                data = await page.evaluate(JS_ASSESS_SECTION)
                
                recommendation = []
                if data['unfilled_inputs'] > 0:
                    recommendation.append(f"Fill {data['unfilled_inputs']} visible inputs.")
                
                if data['actions']:
                    recommendation.append(f"Consider clicking: {', '.join(data['actions'])}.")
                
                if not data['at_bottom']:
                    recommendation.append("Scroll down to see more content.")
                else:
                    recommendation.append("You are at the bottom of the page.")
                
                status = f"Page Status: {data['progress']}% viewed."
                return f"{status}\nAnalysis: {' '.join(recommendation)}"
            except Exception as e:
                return f"Error assessing page: {e}"

        @self.registry.action("Check if an element containing specific text appears selected/active")
        async def check_element_state_by_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            try:
                results = await smart_scan(page, JS_CHECK_TEXT_ELEMENT_STATUS, args=text)
                if not results:
                    return f"No visible elements found containing text: '{text}'"
                
                # Filter for likely selected
                selected = [r for r in results if r['isLikelySelected']]
                if selected:
                    return f"YES. Found {len(selected)} element(s) with text '{text}' that appear selected/active. Details: {json.dumps(selected, indent=2)}"
                else:
                    return f"NO. Found elements with text '{text}', but none appear selected. They might be unselected or use a non-standard state indication."
            except Exception as e:
                return f"Error checking element state: {e}"

        @self.registry.action("Focus on specific content by text (scrolls and highlights)")
        async def focus_and_highlight(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            found = await focus_on_content(page, text)
            if found:
                return f"Focused and highlighted content matching '{text}'."
            return f"Content '{text}' not found."

        @self.registry.action("Click on an element containing specific text")
        async def click_element_by_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()
            try:
                # Strategy 1: Text selector (Playwright's robust text matching)
                loc = page.locator(f"text={text}").first
                if await loc.is_visible():
                    await loc.click()
                    return f"Clicked element with text '{text}'"
                
                return f"Could not find visible element with text '{text}'"
            except Exception as e:
                return f"Error clicking text '{text}': {e}"

        @self.registry.action("Type into an input field identified by nearby text/label")
        async def type_into_element_by_text(browser: BrowserContext, text: str, value: str):
            page = await browser.get_current_page()
            try:
                # 1. Try Label
                try:
                    await page.get_by_label(text).fill(value)
                    return f"Typed '{value}' into field labeled '{text}'"
                except: pass
                
                # 2. Try Placeholder
                try:
                    await page.get_by_placeholder(text).fill(value)
                    return f"Typed '{value}' into field with placeholder '{text}'"
                except: pass
                
                return f"Could not find input field associated with '{text}'"
            except Exception as e:
                return f"Error typing into '{text}': {e}"

        @self.registry.action("Read the full text content of the current page")
        async def read_page_content(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                content = await page.evaluate("document.body.innerText")
                return f"Page Content ({len(content)} chars):\n{content[:5000]}..." # Truncate for log, but agent gets it
            except Exception as e:
                return f"Error reading page content: {e}"

        @self.registry.action("Save the current page text content to a file for review")
        async def save_page_content(browser: BrowserContext, filename: str = "page_content.txt"):
            page = await browser.get_current_page()
            try:
                content = await page.evaluate("document.body.innerText")
                title = await page.title()
                url = page.url
                full_text = f"URL: {url}\nTitle: {title}\n\n{content}"
                
                safe_dir = os.path.abspath("./tmp/downloads")
                os.makedirs(safe_dir, exist_ok=True)
                filepath = os.path.join(safe_dir, os.path.basename(filename))
                
                utils_save_text(filepath, full_text, "w")
                return f"Saved page content to {filepath}"
            except Exception as e:
                return f"Error saving page content: {e}"
