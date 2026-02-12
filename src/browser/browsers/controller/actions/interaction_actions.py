import logging
import os
import json
from browser_use.browser.context import BrowserContext
from src.utils.utils import resolve_file_path
from src.utils.browser_scripts import (
    JS_VERIFY_ELEMENT_STATE,
    JS_SET_DATE_VALUE,
    JS_SET_RANGE_VALUE,
    JS_SET_COLOR_VALUE
)

logger = logging.getLogger(__name__)

class InteractionActionsMixin:
    def __init__(self, *args, **kwargs):
        # This mixin expects to be mixed into a class that has a registry attribute
        # We define it here to satisfy type checkers and ensure modularity
        if not hasattr(self, 'registry'):
            self.registry = None
        if not hasattr(self, 'clipboard_content'):
            self.clipboard_content = ""

    def _register_interaction_actions(self):
        if not self.registry:
            logger.warning("InteractionActionsMixin: No registry found, actions not registered.")
            return

        @self.registry.action("Simulate hovering over an element")
        async def hover_element(browser: BrowserContext, selector: str):
            page = await browser.get_current_page()
            try:
                await page.hover(selector)
                return f"Hovered over element: {selector}"
            except Exception as e:
                return f"Error hovering over element: {e}"

        @self.registry.action("Simulate pressing a specific key combination")
        async def press_key(browser: BrowserContext, key: str):
            page = await browser.get_current_page()
            try:
                await page.keyboard.press(key)
                return f"Pressed key: {key}"
            except Exception as e:
                return f"Error pressing key: {e}"

        @self.registry.action("Upload a file to a specific selector")
        async def upload_file_to_element(browser: BrowserContext, selector: str, filename: str):
            page = await browser.get_current_page()
            
            filepath = resolve_file_path(filename)
            if not filepath:
                return f"File not found: {filename}. Checked common directories (absolute path, downloads, uploads, cwd)."
            
            try:
                await page.set_input_files(selector, filepath)
                return f"Uploaded {os.path.basename(filepath)} to {selector}"
            except Exception as e:
                return f"Error uploading file: {e}"

        @self.registry.action("Upload a file to an input element identified by text/label")
        async def upload_file_by_text(browser: BrowserContext, text: str, filename: str):
            page = await browser.get_current_page()
            
            filepath = resolve_file_path(filename)
            if not filepath:
                return f"File not found: {filename}"

            try:
                #  Try to find the input element associated with the text (Label)
                await page.get_by_label(text).set_input_files(filepath)
                return f"Uploaded {filename} to field labeled '{text}'"
            except Exception as e:
                return f"Error uploading file to '{text}': {e}. Try using a specific selector if possible."

        @self.registry.action("Upload a file by simulating a drag-and-drop event onto a drop zone identified by text")
        async def upload_file_drag_and_drop_by_text(browser: BrowserContext, text: str, filename: str):
            page = await browser.get_current_page()
            
            filepath = resolve_file_path(filename)
            if not filepath:
                return f"File not found: {filename}"

            try:
                # Find the drop zone element by text
                element = page.locator(f"text={text}").first
                if await element.count() == 0:
                    return f"Drop zone text '{text}' not found."

                # Strategy: Look for a file input nearby (child or parent)
                # 1. Check children
                input_loc = element.locator("input[type='file']").first
                if await input_loc.count() > 0:
                    await input_loc.set_input_files(filepath)
                    return f"Uploaded {filename} via hidden input inside '{text}' zone."
                
                # 2. Check parent (often the text is inside a span inside the drop div)
                parent = element.locator("..")
                input_loc = parent.locator("input[type='file']").first
                if await input_loc.count() > 0:
                    await input_loc.set_input_files(filepath)
                    return f"Uploaded {filename} via hidden input in parent of '{text}'."

                return f"Could not find a file input associated with '{text}'. Try using the standard 'upload' action."
            except Exception as e:
                return f"Error in drag-and-drop upload: {e}"

        @self.registry.action("Verify if a file upload was successful by checking page text")
        async def verify_upload(browser: BrowserContext, filename: str):
            page = await browser.get_current_page()
            try:
                # Check if filename appears in body text (common for upload lists)
                found = await page.evaluate(f"() => document.body.innerText.includes('{filename}')")
                if found:
                    return f"Verified: '{filename}' appears on the page."
                return f"Not Verified: '{filename}' not found in page text."
            except Exception as e:
                return f"Error verifying upload: {e}"

        @self.registry.action("Verify the state of an element (checked, visible, enabled, etc.)")
        async def verify_element_state(browser: BrowserContext, selector: str, state: str):
            page = await browser.get_current_page()
            try:
                # Valid states: checked, unchecked, visible, hidden, enabled, disabled, editable, readonly, selected
                result = await page.evaluate(JS_VERIFY_ELEMENT_STATE, [selector, state])
                if not result['found']:
                    return f"Element '{selector}' not found."
                
                if result['match']:
                    return f"Verified: Element '{selector}' is {state}."
                else:
                    return f"Check Failed: Element '{selector}' is NOT {state}."
            except Exception as e:
                return f"Error verifying element state: {e}"

        @self.registry.action("Hover over an element (by text) and click a target (by text)")
        async def hover_and_click_by_text(browser: BrowserContext, hover_text: str, click_text: str):
            page = await browser.get_current_page()
            try:
                # Hover
                hover_el = page.get_by_text(hover_text).first
                if await hover_el.count() > 0:
                    await hover_el.hover()
                else:
                    return f"Hover element with text '{hover_text}' not found."
                
                await page.wait_for_timeout(500)
                
                # Click
                click_el = page.get_by_text(click_text).first
                if await click_el.is_visible():
                    await click_el.click()
                    return f"Hovered '{hover_text}' and clicked '{click_text}'."
                return f"Hovered '{hover_text}', but target '{click_text}' was not visible."
            except Exception as e:
                return f"Error in hover_and_click: {e}"

        @self.registry.action("Set the value of a date input field identified by label")
        async def set_date_by_label(browser: BrowserContext, label: str, date_iso: str):
            page = await browser.get_current_page()
            try:
                element = page.get_by_label(label).first
                if await element.count() == 0:
                    return f"Date input with label '{label}' not found."
                
                result = await element.evaluate(JS_SET_DATE_VALUE, date_iso)
                if result.get('success'):
                    return f"Set date '{date_iso}' on input labeled '{label}'."
                return f"Failed to set date: {result.get('error')}"
            except Exception as e:
                return f"Error setting date: {e}"

        @self.registry.action("Set the value of a range/slider input identified by label")
        async def set_range_by_label(browser: BrowserContext, label: str, value: str):
            page = await browser.get_current_page()
            try:
                element = page.get_by_label(label).first
                if await element.count() == 0:
                    return f"Range input with label '{label}' not found."
                
                result = await element.evaluate(JS_SET_RANGE_VALUE, value)
                if result.get('success'):
                    return f"Set range '{label}' to '{value}'."
                return f"Failed to set range: {result.get('error')}"
            except Exception as e:
                return f"Error setting range: {e}"

        @self.registry.action("Set the value of a color picker input identified by label")
        async def set_color_by_label(browser: BrowserContext, label: str, color: str):
            page = await browser.get_current_page()
            try:
                element = page.get_by_label(label).first
                if await element.count() == 0:
                    return f"Color input with label '{label}' not found."
                
                # Ensure hex format
                result = await element.evaluate(JS_SET_COLOR_VALUE, color)
                if result.get('success'):
                    return f"Set color '{label}' to '{color}'."
                return f"Failed to set color: {result.get('error')}"
            except Exception as e:
                return f"Error setting color: {e}"

        @self.registry.action("Handle the next browser dialog (alert/confirm/prompt)")
        async def handle_next_dialog(browser: BrowserContext, action: str = "accept", prompt_text: str = ""):
            page = await browser.get_current_page()
            
            async def handle_dialog(dialog):
                if action == "accept":
                    await dialog.accept(prompt_text)
                else:
                    await dialog.dismiss()
            
            page.once("dialog", handle_dialog)
            return f"Set up handler to {action} the next dialog."

        @self.registry.action("Select an option from a dropdown element")
        async def select_dropdown_option(browser: BrowserContext, selector: str, value: str):
            page = await browser.get_current_page()
            try:
                await page.select_option(selector, value)
                return f"Selected option '{value}' in {selector}"
            except Exception as e:
                return f"Error selecting option: {e}"

        @self.registry.action("Drag and drop an element to a target location")
        async def drag_and_drop(browser: BrowserContext, source_selector: str, target_selector: str):
            page = await browser.get_current_page()
            try:
                await page.drag_and_drop(source_selector, target_selector)
                return f"Dragged {source_selector} to {target_selector}"
            except Exception as e:
                return f"Error dragging and dropping: {e}"

        @self.registry.action("Execute JavaScript in a specific iframe")
        async def execute_js_in_iframe(browser: BrowserContext, frame_index: int, code: str):
            page = await browser.get_current_page()
            frames = page.frames
            if frame_index < 0 or frame_index >= len(frames):
                return f"Frame index {frame_index} out of bounds."
            try:
                result = await frames[frame_index].evaluate(code)
                return f"Result: {json.dumps(result, default=str)}"
            except Exception as e:
                return f"Error executing JS: {e}"

        @self.registry.action("Copy text to internal clipboard")
        async def copy_to_clipboard(text: str):
            self.clipboard_content = text
            return "Text copied to internal clipboard."

        @self.registry.action("Paste text from internal clipboard to an element")
        async def paste_from_clipboard(browser: BrowserContext, selector: str):
            if not self.clipboard_content:
                return "Clipboard is empty."
            page = await browser.get_current_page()
            try:
                await page.fill(selector, self.clipboard_content)
                return f"Pasted content to {selector}"
            except Exception as e:
                return f"Error pasting content: {e}"

        @self.registry.action("Read internal clipboard content")
        async def read_clipboard():
            return self.clipboard_content if self.clipboard_content else "Clipboard is empty."

        @self.registry.action("Attempt to solve CAPTCHAs (reCAPTCHA, Turnstile, or generic checkboxes). Use this if the page asks to 'Verify you are human'.")
        async def solve_captcha(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                # Google reCAPTCHA
                recaptcha_frame = page.frame_locator("iframe[src*='google.com/recaptcha'][src*='anchor']")
                if await recaptcha_frame.locator(".recaptcha-checkbox-border").count() > 0:
                    await recaptcha_frame.locator(".recaptcha-checkbox-border").click()
                    await page.wait_for_timeout(2000)
                    return "Clicked Google reCAPTCHA checkbox. Please verify if solved."
                
                # Cloudflare Turnstile
                cf_frame = page.frame_locator("iframe[src*='challenges.cloudflare.com']")
                if await cf_frame.locator("input[type='checkbox']").count() > 0:
                    await cf_frame.locator("input[type='checkbox']").click()
                    await page.wait_for_timeout(2000)
                    return "Clicked Cloudflare Turnstile checkbox."
                
                # Generic "I am human" buttons
                human_btns = page.locator("button:has-text('I am human'), button:has-text('Verify you are human')")
                if await human_btns.count() > 0:
                    await human_btns.first.click()
                    return "Clicked generic human verification button."

                return "No simple CAPTCHA found. If a complex CAPTCHA is present, please ask the user for help."
            except Exception as e:
                return f"Error attempting to solve CAPTCHA: {e}"

        @self.registry.action("Post content to Yellowdig (handles word count and posting)")
        async def post_to_yellowdig(browser: BrowserContext, content: str):
            """
            Posts content to Yellowdig. Validates word count (>=40).
            Ensures content is typed into the editor and the Post button is clicked.
            """
            # 1. Validate Word Count & Quality Heuristics
            word_count = len(content.split())
            if word_count < 40:
                return f"Error: Content is too short ({word_count} words). Yellowdig requires 40+ words for points. Please expand your post to be formal and professional."

            page = await browser.get_current_page()
            try:
                # 2. Locate Editor (Yellowdig often uses Quill or contenteditable divs)
                editor_locators = [
                    ".ql-editor", 
                    "div[contenteditable='true']", 
                    "textarea[placeholder*='Post']",
                    "textarea[placeholder*='Share']"
                ]
                
                editor = None
                for selector in editor_locators:
                    loc = page.locator(selector).first
                    if await loc.is_visible():
                        editor = loc
                        break
                
                if not editor:
                    return "Error: Could not find a visible post editor. Ensure you have clicked 'Create' or 'Post' to open the dialog first."

                # 3. Input Content
                await editor.click()
                await page.wait_for_timeout(500)
                await editor.fill(content)
                
                # 4. Click Post
                post_btn = page.locator("button:has-text('Post'), button:has-text('Create Post')").first
                
                if await post_btn.is_visible():
                    if await post_btn.is_disabled():
                        return "Error: 'Post' button is disabled. You may need to select a Topic or Tag manually before I can post."
                    await post_btn.click()
                    await page.wait_for_timeout(3000) # Wait for post to process
                    return f"Successfully posted to Yellowdig. Word count: {word_count}."
                
                return "Error: Content typed, but 'Post' button not found."

            except Exception as e:
                return f"Error executing post_to_yellowdig: {e}"
