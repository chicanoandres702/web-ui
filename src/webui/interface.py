import warnings
# Suppress Gradio deprecation warnings immediately
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

import gradio as gr
from functools import partial

from src.webui.webui_manager import WebuiManager
from src.webui.components.agent_settings_tab import create_agent_settings_tab
from src.webui.components.browser_settings_tab import create_browser_settings_tab
from src.webui.components.browser_use_agent_tab import create_browser_use_agent_tab
from src.webui.components.deep_research_agent_tab import create_deep_research_agent_tab
from src.webui.components.load_save_config_tab import create_load_save_config_tab

theme_map = {
    "Default": gr.themes.Default(),
    "Soft": gr.themes.Soft(),
    "Monochrome": gr.themes.Monochrome(),
    "Glass": gr.themes.Glass(),
    "Origin": gr.themes.Origin(),
    "Citrus": gr.themes.Citrus(),
    "Ocean": gr.themes.Ocean(),
    "Base": gr.themes.Base()
}


def create_ui(theme_name="Ocean"):
    css = """
    :root {
        --primary-color: #3b82f6;
        --secondary-color: #64748b;
        --background-color: #f8fafc;
        --surface-color: #ffffff;
        --text-color: #1e293b;
        --border-radius: 12px;
    }

    .dark {
        --primary-color: #3b82f6;
        --secondary-color: #94a3b8;
        --background-color: #0f172a;
        --surface-color: #1e293b;
        --text-color: #f8fafc;
    }
    
    .gradio-container {
        width: 95% !important; 
        max-width: 1600px !important; 
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 20px !important;
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .header-text {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: var(--surface-color);
        border-radius: var(--border-radius);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .header-text h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* Tabs Styling */
    .tabs {
        background: var(--surface-color);
        border-radius: var(--border-radius);
        padding: 10px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Chatbot Styling */
    #browser_use_chatbot {
        height: 600px !important;
        border-radius: var(--border-radius);
        border: 1px solid #e2e8f0;
        background: #ffffff;
    }
    
    /* Input Area */
    #user_input textarea {
        border-radius: var(--border-radius);
        border: 1px solid #e2e8f0;
        padding: 12px;
        font-size: 1rem;
    }
    
    /* Buttons */
    button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Browser View */
    #browser_view {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Agent Thought Visualization */
    .agent-thought-container {
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 12px;
        background-color: var(--background-fill-secondary);
        margin-bottom: 10px;
    }
    .thought-badge {
        background-color: var(--primary-color);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 8px;
        text-transform: uppercase;
    }
    .retry-badge {
        background-color: #8b5cf6; /* Violet */
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 8px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 5px;
    }
    """

    # dark mode by default
    js_func = """
    function refresh() {
        document.body.classList.add('dark');
    }
    """

    ui_manager = WebuiManager()

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css, js=js_func,
    ) as demo:
        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown(
                    """
                    # 🌐 Browser Use WebUI
                    ### Control your browser with AI assistance
                    """,
                    elem_classes=["header-text"],
                )
            with gr.Column(scale=1, min_width=100):
                theme_toggle = gr.Button("🌓 Theme", variant="secondary")
                theme_toggle.click(None, None, None, js="() => { document.body.classList.toggle('dark'); }")

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings"):
                create_agent_settings_tab(ui_manager)

            with gr.TabItem("🌐 Browser Settings"):
                create_browser_settings_tab(ui_manager)

            with gr.TabItem("🤖 Run Agent"):
                create_browser_use_agent_tab(ui_manager)

            with gr.TabItem("🎁 Agent Marketplace"):
                gr.Markdown(
                    """
                    ### Agents built on Browser-Use
                    """,
                    elem_classes=["tab-header-text"],
                )
                with gr.Tabs():
                    with gr.TabItem("Deep Research"):
                        create_deep_research_agent_tab(ui_manager)

            with gr.TabItem("📁 Load & Save Config"):
                create_load_save_config_tab(ui_manager)

        # Auto-save binding
        for comp in ui_manager.get_components():
            if not isinstance(comp, (gr.Button, gr.File)) and getattr(comp, "interactive", True):
                if hasattr(comp, "change"):
                    comp.change(fn=partial(ui_manager.update_parameter, comp), inputs=[comp])

        # Load last config on startup
        demo.load(fn=ui_manager.load_last_config, outputs=ui_manager.get_components())

    return demo
