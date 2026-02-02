import asyncio
import os
import sys
from pprint import pprint
from dotenv import load_dotenv

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.utils import llm_provider

load_dotenv()

async def test_custom_tools():
    """
    Test script to verify the functionality of CustomController tools.
    """
    print("--- Starting Custom Tools Verification ---")

    # 1. Setup LLM
    # Using OpenAI by default, but you can switch to others configured in llm_provider
    try:
        llm = llm_provider.get_llm_model(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.0,  # Low temperature for deterministic tool usage
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Please ensure your .env file has the correct API keys.")
        return

    # 2. Setup Controller with new tools
    controller = CustomController()
    print("Controller initialized with custom tools.")

    # 3. Setup Browser
    # Set headless=False to visually see the browser actions
    browser = CustomBrowser(
        config=BrowserConfig(
            headless=False,
        )
    )
    
    # Create context with specific window size
    browser_context = await browser.new_context(
        config=BrowserContextConfig(
            window_width=1280,
            window_height=1100,
            save_downloads_path="./tmp/downloads"
        )
    )

    # 4. Define a comprehensive task to exercise the new tools
    # We explicitly instruct the agent to use specific tools in a logical sequence.
    task = (
        "Navigate to 'https://www.example.com'. "
        "1. Enable console log capture immediately to catch any events. "
        "2. Scroll and assess the page to ensure all content is loaded. "
        "3. Get the page heading structure to understand the layout. "
        "4. Extract all hyperlinks from the page. "
        "5. Inspect the HTML details of the main 'h1' element (or the first visible header). "
        "6. Get the current cookies for this session. "
        "7. Save the extracted links to a file named 'test_links.txt'. "
        "8. Finally, retrieve the captured console logs."
    )

    # 5. Create Agent
    agent = BrowserUseAgent(
        task=task,
        llm=llm,
        browser=browser,
        browser_context=browser_context,
        controller=controller,
        use_vision=True,
    )

    try:
        print(f"\nExecuting Task: {task}\n")
        history = await agent.run(max_steps=15)
        
        print("\n--- Final Result ---")
        pprint(history.final_result())
        
        print("\n--- Tool Execution Summary ---")
        tool_usage = []
        for i, item in enumerate(history.history):
            if item.model_output and item.model_output.action:
                for action in item.model_output.action:
                    # Extract tool name from the action dictionary
                    tool_name = next(iter(action.keys()))
                    tool_usage.append(tool_name)
                    print(f"Step {i+1}: {tool_name}")
        
        # Verification checks
        expected_tools = [
            'enable_console_log_capture', 
            'scroll_and_assess_page', 
            'get_page_structure', 
            'extract_page_links', 
            'inspect_element_details', 
            'get_cookies', 
            'save_text_to_file', 
            'get_console_logs'
        ]
        
        print("\n--- Verification ---")
        for tool in expected_tools:
            if tool in tool_usage:
                print(f"✅ {tool} was called.")
            else:
                print(f"❌ {tool} was NOT called.")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing browser...")
        await browser_context.close()
        await browser.close()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(test_custom_tools())
