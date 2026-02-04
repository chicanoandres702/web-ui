import os
import gradio as gr
import logging
from gradio.components import Component

from src.webui.webui_manager import WebuiManager
from src.utils import config
from src.utils.utils import str_to_bool
from src.webui.components.browser_logic import (
    close_browser,
    clear_browser_session,
    get_session_info,
    update_env_var
)

logger = logging.getLogger(__name__)

def create_browser_settings_tab(webui_manager: WebuiManager):
    """
    Creates a browser settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    # --- Group 1: Browser Configuration ---
    with gr.Group():
        gr.Markdown("### 🌐 Browser Configuration")
        with gr.Row():
            browser_binary_path = gr.Textbox(
                label="Browser Binary Path",
                lines=1,
                interactive=True,
                placeholder="Auto-detected if empty. e.g. 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'",
                scale=3
            )
            use_own_browser = gr.Checkbox(
                label="Use Own Browser",
                value=str_to_bool(os.getenv("USE_OWN_BROWSER", "false")),
                info="Use system Chrome/Edge instead of bundled Chromium. Auto-detects path if empty.",
                interactive=True,
                scale=1
            )
            chrome_profile_name = gr.Textbox(
                label="Chrome Profile Name",
                info="Creates a separate session folder (e.g. 'Profile 1') to allow parallel agents.",
                interactive=True,
                value=os.getenv("CHROME_PROFILE_NAME", "")
            )
            extra_browser_args = gr.Textbox(
                label="Extra Browser Args",
                info="Space-separated args e.g. --ignore-certificate-errors",
                interactive=True,
                value=os.getenv("EXTRA_BROWSER_ARGS", "")
            )

    # --- Group 2: Session Persistence ---
    with gr.Group():
        gr.Markdown("### 💾 Session Persistence")
        enable_persistent_session = gr.Checkbox(
            label="Enable Persistent Session",
            value=str_to_bool(os.getenv("ENABLE_PERSISTENT_SESSION", "true")),
            info=f"Retain cookies, logins, and session data between runs. Uses {config.DEFAULT_BROWSER_SESSION_DIR} by default.",
            interactive=True
        )
        
        with gr.Column(visible=True) as persistence_options:
            browser_user_data_dir = gr.Textbox(
                label="Browser User Data Dir",
                lines=1,
                interactive=True,
                placeholder=f"Defaults to {config.DEFAULT_BROWSER_SESSION_DIR} if empty",
            )
            with gr.Row():
                verify_session_btn = gr.Button("🔍 Verify Session Storage", size="sm", scale=0)
                clear_session_btn = gr.Button("🗑️ Clear Session", size="sm", scale=0, variant="stop")
                session_status = gr.Textbox(show_label=False, interactive=False, lines=1, scale=4, placeholder="Click verify to check storage usage...")

    # --- Group 3: Run Options ---
    with gr.Group():
        gr.Markdown("### ⚙️ Run Options")
        with gr.Row():
            headless = gr.Checkbox(
                label="Headless Mode",
                value=False,
                info="Run browser without GUI",
                interactive=True
            )
            disable_security = gr.Checkbox(
                label="Disable Security",
                value=False,
                info="Disable browser security",
                interactive=True
            )
            keep_browser_open = gr.Checkbox(
                label="Keep Browser Open",
                value=str_to_bool(os.getenv("KEEP_BROWSER_OPEN", "true")),
                info="Keep Browser Open between Tasks",
                interactive=True
            )

    # --- Group 4: Viewport & Connection ---
    with gr.Group():
        gr.Markdown("### 🖥️ Viewport & Connection")
        with gr.Row():
            window_w = gr.Number(label="Window Width", value=1280, interactive=True, minimum=128, precision=0)
            window_h = gr.Number(label="Window Height", value=1100, interactive=True, minimum=128, precision=0)
        
        with gr.Row():
            cdp_url = gr.Textbox(
                label="CDP URL",
                value=os.getenv("BROWSER_CDP", None),
                info="CDP URL for browser remote debugging",
                interactive=True,
            )
            wss_url = gr.Textbox(label="WSS URL", info="WSS URL for remote debugging", interactive=True)

    # --- Group 5: Recording & Debugging ---
    with gr.Group():
        gr.Markdown("### 📹 Recording & Debugging")
        with gr.Row():
            save_recording_path = gr.Textbox(
                label="Recording Path",
                placeholder="e.g. ./tmp/record_videos",
                info="Path to save browser recordings",
                interactive=True,
            )

            save_trace_path = gr.Textbox(
                label="Trace Path",
                placeholder="e.g. ./tmp/traces",
                info="Path to save Agent traces",
                interactive=True,
            )

        with gr.Row():
            save_agent_history_path = gr.Textbox(
                label="Agent History Save Path",
                value="./tmp/agent_history",
                info="Specify the directory where agent history should be saved.",
                interactive=True,
            )
            save_download_path = gr.Textbox(
                label="Save Directory for browser downloads",
                value="./tmp/downloads",
                info="Specify the directory where downloaded files should be saved.",
                interactive=True,
            )
    tab_components.update(
        dict(
            browser_binary_path=browser_binary_path,
            browser_user_data_dir=browser_user_data_dir,
            use_own_browser=use_own_browser,
            chrome_profile_name=chrome_profile_name,
            extra_browser_args=extra_browser_args,
            enable_persistent_session=enable_persistent_session,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            save_recording_path=save_recording_path,
            save_trace_path=save_trace_path,
            save_agent_history_path=save_agent_history_path,
            save_download_path=save_download_path,
            cdp_url=cdp_url,
            wss_url=wss_url,
            window_h=window_h,
            window_w=window_w,
            verify_session_btn=verify_session_btn,
            session_status=session_status,
        )
    )
    webui_manager.add_components("browser_settings", tab_components)

    async def close_wrapper():
        """Wrapper for handle_clear."""
        await close_browser(webui_manager)

    headless.change(close_wrapper)
    keep_browser_open.change(close_wrapper)
    disable_security.change(close_wrapper)
    use_own_browser.change(close_wrapper)
    enable_persistent_session.change(close_wrapper)
    
    # Toggle persistence options visibility
    enable_persistent_session.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[enable_persistent_session],
        outputs=[persistence_options]
    )
    
    verify_session_btn.click(fn=get_session_info, inputs=[browser_user_data_dir], outputs=[session_status])

    async def clear_session_wrapper(user_data_dir):
        return await clear_browser_session(webui_manager, user_data_dir)

    clear_session_btn.click(fn=clear_session_wrapper, inputs=[browser_user_data_dir], outputs=[session_status])

    # Bind environment variable updates
    chrome_profile_name.change(fn=lambda x: update_env_var("CHROME_PROFILE_NAME", x), inputs=[chrome_profile_name])
    extra_browser_args.change(fn=lambda x: update_env_var("EXTRA_BROWSER_ARGS", x), inputs=[extra_browser_args])
