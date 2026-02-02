import pyperclip
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar, List
from datetime import datetime
from urllib.parse import urlparse
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from browser_use.controller.registry.service import Registry, RegisteredAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
import inspect
import asyncio
import os
import json
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.agent.views import ActionModel, ActionResult

from src.utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools

from browser_use.utils import time_execution_sync, time_execution_async
from src.utils.utils import save_text_to_file_async, save_to_knowledge_base_file

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 confirm_action_callback: Optional[Union[Callable[[str, Dict[str, Any], BrowserContext], str], Callable[
                     [str, Dict[str, Any], BrowserContext], Awaitable[str]]]] = None,
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.confirm_action_callback = confirm_action_callback
        self.mcp_client = None
        self.mcp_server_config = None
        self.memory_file = None
        self.history: List[Dict[str, Any]] = []
        self.fast_mode = False
        self.require_confirmation = False
        self.critical_actions = ["click_element", "input_text", "go_to_url", "upload_file", "send_keys", "scroll_to_text"]

    def set_fast_mode(self, enabled: bool):
        self.fast_mode = enabled

    def set_require_confirmation(self, enabled: bool):
        self.require_confirmation = enabled

    def set_memory_file(self, path: Optional[str]):
        self.memory_file = path

    async def _read_text_from_file(self, path: str) -> str:
        """Helper to safely read text from a file asynchronously."""
        def _read():
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            return ""
        return await asyncio.to_thread(_read)

    def reset_loop_history(self):
        self.history = []

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. However, if you encounter a definitive blocker "
            "that prevents you from proceeding independently – such as needing credentials you don't possess, "
            "requiring subjective human judgment, needing a physical action performed, encountering complex CAPTCHAs, "
            "or facing limitations in your capabilities – you must request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(extracted_content="Human cannot help you. Please try another way.",
                                    include_in_memory=True)

        @self.registry.action(
            'Upload file to interactive element with file path ',
        )
        async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')

            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')

            dom_el = await browser.get_dom_element_by_index(index)

            file_upload_dom_el = dom_el.get_file_upload_element()

            if file_upload_dom_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)

            if file_upload_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

        @self.registry.action("Save important information to long-term memory.")
        async def save_to_memory(text: str):
            if self.memory_file:
                try:
                    entry = f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
                    await save_text_to_file_async(self.memory_file, entry, mode="a")
                    msg = f"Saved to memory: {text}"
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                except Exception as e:
                    msg = f"Failed to save to memory: {str(e)}"
                    logger.error(msg)
                    return ActionResult(error=msg)
            else:
                return ActionResult(error="Memory file not configured.")

        @self.registry.action("Save structured knowledge to a specific topic database. The agent decides the topic name.")
        async def save_to_knowledge_base(text: str, topic: str):
            """
            Saves text to a specific knowledge base file based on the topic.
            """
            try:
                filepath = await asyncio.to_thread(save_to_knowledge_base_file, text, topic, self.memory_file or "./tmp/memory/memory.txt")
                msg = f"Saved information to knowledge base '{topic}': {text[:50]}..."
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f"Failed to save to knowledge base '{topic}': {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("Save specific knowledge about how to navigate or use a website. Use this when you learn how to bypass a popup, find a specific resource, or navigate a complex menu.")
        async def save_site_knowledge(knowledge: str, url: str):
            """
            Saves knowledge about a specific website.
            """
            if self.memory_file:
                base_dir = os.path.dirname(os.path.abspath(self.memory_file))
            else:
                base_dir = os.path.abspath("./tmp/memory")
            
            sites_dir = os.path.join(base_dir, "sites")
            os.makedirs(sites_dir, exist_ok=True)
            
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                if not domain:
                    return ActionResult(error="Invalid URL provided.")
                
                filename = f"{domain}.md"
                filepath = os.path.join(sites_dir, filename)
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                entry = f"\n## [{timestamp}]\n{knowledge}\n"
                await save_text_to_file_async(filepath, entry, mode="a")
                    
                msg = f"Saved site knowledge for '{domain}': {knowledge[:50]}..."
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f"Failed to save site knowledge: {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("Retrieve stored knowledge about a specific website. Use this when visiting a new domain to check for past learnings.")
        async def get_site_knowledge(url: str):
            """
            Retrieves knowledge about a specific website.
            """
            if self.memory_file:
                base_dir = os.path.dirname(os.path.abspath(self.memory_file))
            else:
                base_dir = os.path.abspath("./tmp/memory")
            
            sites_dir = os.path.join(base_dir, "sites")
            
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                if not domain:
                    return ActionResult(error="Invalid URL provided.")
                
                filename = f"{domain}.md"
                filepath = os.path.join(sites_dir, filename)
                
                if os.path.exists(filepath):
                    content = await self._read_text_from_file(filepath)
                    
                    msg = f"Knowledge for '{domain}':\n{content}"
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    return ActionResult(extracted_content=f"No specific knowledge found for '{domain}'.", include_in_memory=True)
            except Exception as e:
                msg = f"Failed to get site knowledge: {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("List all available knowledge base topics.")
        async def list_knowledge_topics():
            """
            Lists all available knowledge base topics.
            """
            if self.memory_file:
                base_dir = os.path.dirname(os.path.abspath(self.memory_file))
            else:
                base_dir = os.path.abspath("./tmp/memory")
            
            def _list():
                topics = []
                if os.path.exists(base_dir):
                    for f in os.listdir(base_dir):
                        if f.startswith("kb_") and f.endswith(".md"):
                            topics.append(f[3:-3]) # remove kb_ and .md
                return topics
            
            topics = await asyncio.to_thread(_list)
            return ActionResult(extracted_content=f"Available topics: {', '.join(topics)}", include_in_memory=True)

        @self.registry.action("Search through all knowledge bases for a specific query.")
        async def search_knowledge_base(query: str):
            """
            Searches for a query string across all knowledge base files.
            """
            if self.memory_file:
                base_dir = os.path.dirname(os.path.abspath(self.memory_file))
            else:
                base_dir = os.path.abspath("./tmp/memory")
            
            def _search():
                results = []
                if os.path.exists(base_dir):
                    for root, _, files in os.walk(base_dir):
                        for file in files:
                            if file.endswith(".md") or file.endswith(".txt"):
                                path = os.path.join(root, file)
                                try:
                                    with open(path, "r", encoding="utf-8") as f:
                                        content = f.read()
                                        if query.lower() in content.lower():
                                            # Extract a snippet around the match
                                            idx = content.lower().find(query.lower())
                                            start = max(0, idx - 100)
                                            end = min(len(content), idx + 200)
                                            snippet = content[start:end].replace("\n", " ")
                                            results.append(f"File: {file}\nSnippet: ...{snippet}...")
                                except Exception:
                                    continue
                return results

            found = await asyncio.to_thread(_search)
            if found:
                return ActionResult(extracted_content="\n\n".join(found), include_in_memory=True)
            else:
                return ActionResult(extracted_content="No matches found in knowledge bases.", include_in_memory=True)

        @self.registry.action("Archive the full content of the current page to a local file for permanent storage.")
        async def archive_current_page(browser: BrowserContext):
            """
            Saves the full text content of the current page to an archive file.
            """
            if self.memory_file:
                base_dir = os.path.dirname(os.path.abspath(self.memory_file))
            else:
                base_dir = os.path.abspath("./tmp/memory")
            
            archive_dir = os.path.join(base_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            
            try:
                page = await browser.get_current_page()
                title = await page.title()
                content = await page.evaluate("document.body.innerText")
                url = page.url
                
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')[:50]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{safe_title}.txt"
                filepath = os.path.join(archive_dir, filename)
                
                header = f"URL: {url}\nTitle: {title}\nDate: {datetime.now()}\n\n"
                await save_text_to_file_async(filepath, header + content)
                
                msg = f"Archived page '{title}' to {filename}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f"Failed to archive page: {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("Detect and close/remove common chat widgets and overlays.")
        async def close_chat_widget(browser: BrowserContext):
            """
            Attempts to close or remove chat widgets and overlays using common selectors and JavaScript.
            """
            try:
                page = await browser.get_current_page()
                # JavaScript to find and remove/close common chat widgets
                js_script = """
                () => {
                    const selectors = [
                        'iframe[title*="chat"]',
                        'iframe[title*="Chat"]',
                        'iframe[src*="intercom"]',
                        'iframe[src*="drift"]',
                        'iframe[src*="crisp"]',
                        'iframe[src*="zendesk"]',
                        'div[id*="intercom"]',
                        'div[class*="intercom"]',
                        'div[id*="drift"]',
                        'div[class*="drift"]',
                        'div[id*="crisp"]',
                        'div[class*="crisp"]',
                        'div[id*="zendesk"]',
                        'div[class*="zendesk"]',
                        '#hubspot-messenger-iframe',
                        'button[aria-label*="close chat"]',
                        'button[aria-label*="Close Chat"]'
                    ];
                    
                    let count = 0;
                    selectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            el.remove();
                            count++;
                        });
                    });
                    return count;
                }
                """
                removed_count = await page.evaluate(js_script)
                msg = f"Removed {removed_count} chat widget elements."
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f"Failed to remove chat widgets: {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg)

        @self.registry.action("Scroll down the page until the specified text is visible. Use this to find specific items. Returns success if found.")
        async def scroll_to_text(text: str, browser: BrowserContext):
            page = await browser.get_current_page()
            delay_time = 50 if self.fast_mode else 200
            try:
                found = await page.evaluate("""
                    async ({text, delay_time}) => {
                        const delay = ms => new Promise(res => setTimeout(res, ms));
                        const maxScrolls = 20;
                        for (let i = 0; i < maxScrolls; i++) {
                            const xpath = `//*[contains(text(), '${text.replace("'", "\\'")}')]`;
                            const matchingElement = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            
                            if (matchingElement && matchingElement.offsetParent !== null) {
                                matchingElement.scrollIntoView({behavior: "auto", block: "center"});
                                return true;
                            }
                            
                            window.scrollBy(0, window.innerHeight * 0.7);
                            await delay(delay_time);
                            
                            if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight) {
                                break;
                            }
                        }
                        return false;
                    }
                """, {"text": text, "delay_time": delay_time})
                
                if found:
                    return ActionResult(extracted_content=f"Scrolled to find text: '{text}'")
                else:
                    return ActionResult(error=f"Could not find text '{text}' even after scrolling.")
            except Exception as e:
                return ActionResult(error=f"Failed to scroll to text: {str(e)}")

        @self.registry.action("Scroll down the page to see more content. Use this frequently to explore the page.")
        async def scroll_down(browser: BrowserContext):
            page = await browser.get_current_page()
            sleep_time = 0.1 if self.fast_mode else 0.3
            behavior = "auto" if self.fast_mode else "smooth"
            await page.evaluate(f"window.scrollBy({{top: window.innerHeight * 0.75, behavior: '{behavior}'}})")
            await asyncio.sleep(sleep_time)
            return ActionResult(extracted_content="Scrolled down.")

        @self.registry.action("Scroll up the page to see previous content.")
        async def scroll_up(browser: BrowserContext):
            page = await browser.get_current_page()
            sleep_time = 0.1 if self.fast_mode else 0.3
            behavior = "auto" if self.fast_mode else "smooth"
            await page.evaluate(f"window.scrollBy({{top: -window.innerHeight * 0.75, behavior: '{behavior}'}})")
            await asyncio.sleep(sleep_time)
            return ActionResult(extracted_content="Scrolled up.")

        @self.registry.action("Get the full text content of the page. Useful for reading long articles or quizzes.")
        async def get_full_page_text(browser: BrowserContext):
            page = await browser.get_current_page()
            text = await page.evaluate("document.body.innerText")
            return ActionResult(extracted_content=text)

    @time_execution_async('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: Optional[BrowserContext] = None,
            #
            page_extraction_llm: Optional[BaseChatModel] = None,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            #
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            # Loop detection
            current_action_dict = action.model_dump(exclude_unset=True)
            self.history.append(current_action_dict)
            if len(self.history) > 10:
                self.history.pop(0)

            if len(self.history) >= 3:
                last_three = self.history[-3:]
                if all(x == last_three[0] for x in last_three):
                    action_name = list(current_action_dict.keys())[0] if current_action_dict else "unknown"
                    
                    if "scroll" in action_name:
                        if len(self.history) >= 20:
                            last_twenty = self.history[-20:]
                            if all(x == last_twenty[0] for x in last_twenty):
                                return ActionResult(error=f"Loop detected: Action '{action_name}' repeated 20 times. Please try a different strategy.")
                    else:
                        return ActionResult(error=f"Loop detected: Action '{action_name}' repeated 3 times. Please try a different strategy.")

            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    if action_name.startswith("mcp"):
                        # this is a mcp tool
                        logger.debug(f"Invoke MCP tool: {action_name}")
                        mcp_tool = self.registry.registry.actions.get(action_name).function
                        result = await mcp_tool.ainvoke(params)
                    else:
                        # Human in the Loop: Confirmation (Moved inside loop to allow modification)
                        if self.require_confirmation and self.confirm_action_callback and action_name in self.critical_actions:
                            if inspect.iscoroutinefunction(self.confirm_action_callback):
                                response = await self.confirm_action_callback(action_name, params, browser_context)
                            else:
                                response = self.confirm_action_callback(action_name, params, browser_context)
                            
                            if response.lower() in ["no", "skip", "reject", "stop"]:
                                return ActionResult(error=f"Action '{action_name}' rejected by user.")
                            
                            if response.lower().startswith("modify:"):
                                try:
                                    mod_data = response[7:].strip()
                                    new_params = json.loads(mod_data)
                                    if isinstance(params, dict):
                                        params.update(new_params)
                                        logger.info(f"Action '{action_name}' modified. New params: {params}")
                                    else:
                                        return ActionResult(error=f"Cannot modify action '{action_name}': params is not a dictionary.")
                                except Exception as e:
                                    return ActionResult(error=f"Failed to modify action: {e}")

                        result = await self.registry.execute_action(
                            action_name,
                            params,
                            browser=browser_context,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            context=context,
                        )

                    if action_name == "go_to_url" and self.memory_file:
                        try:
                            url = params.get('url')
                            if url:
                                domain = urlparse(url).netloc.replace("www.", "")
                                if domain:
                                    base_dir = os.path.dirname(os.path.abspath(self.memory_file))
                                    sites_dir = os.path.join(base_dir, "sites")
                                    filename = f"{domain}.md"
                                    filepath = os.path.join(sites_dir, filename)
                                    
                                    if os.path.exists(filepath):
                                        site_knowledge = await self._read_text_from_file(filepath)
                                        
                                        knowledge_msg = f"\n\n[Auto-Loaded Site Knowledge for {domain}]:\n{site_knowledge}"
                                        
                                        if isinstance(result, ActionResult):
                                            result.extracted_content = (result.extracted_content or "") + knowledge_msg
                                        elif isinstance(result, str):
                                            result += knowledge_msg
                                        
                                        logger.info(f"Auto-loaded knowledge for {domain}")
                        except Exception as e:
                            logger.error(f"Failed to auto-load site knowledge: {e}")

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        self.mcp_server_config = mcp_server_config
        if self.mcp_server_config:
            self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
            self.register_mcp_tools()

    def register_mcp_tools(self):
        """
        Register the MCP tools used by this controller.
        """
        if self.mcp_client:
            for server_name in self.mcp_client.server_name_to_tools:
                for tool in self.mcp_client.server_name_to_tools[server_name]:
                    tool_name = f"mcp.{server_name}.{tool.name}"
                    self.registry.registry.actions[tool_name] = RegisteredAction(
                        name=tool_name,
                        description=tool.description,
                        function=tool,
                        param_model=create_tool_param_model(tool),
                    )
                    logger.info(f"Add mcp tool: {tool_name}")
                logger.debug(
                    f"Registered {len(self.mcp_client.server_name_to_tools[server_name])} mcp tools for {server_name}")
        else:
            logger.warning(f"MCP client not started.")

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
