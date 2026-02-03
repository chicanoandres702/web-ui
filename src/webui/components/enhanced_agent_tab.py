import gradio as gr
import asyncio
from src.webui.webui_manager import WebuiManager
from src.webui.components.shared import get_agent_settings_values, get_browser_settings_values, initialize_agent_llms, safe_execution, get_valid_input_components
from src.webui.components.agent_logic import initialize_browser_infrastructure, configure_controller
from src.agent.deep_research.enhanced_agent import EnhancedDeepResearchAgent
import logging

logger = logging.getLogger(__name__)

async def run_enhanced_agent(webui_manager: WebuiManager, goal: str, url: str, components):
    """Handler for running the enhanced agent."""
    
    # 1. Setup Infrastructure
    agent_settings = get_agent_settings_values(webui_manager, components)
    browser_settings = get_browser_settings_values(webui_manager, components)
    
    # Initialize LLM
    llm, _, _, _, _ = await initialize_agent_llms(agent_settings)
    if not llm:
        yield "LLM initialization failed. Please check Agent Settings."
        return
    
    # Initialize Browser
    await initialize_browser_infrastructure(webui_manager, browser_settings)
    
    # Initialize Controller
    controller = await configure_controller(webui_manager, agent_settings, None, None, {})
    
    # 2. Initialize Agent
    agent = EnhancedDeepResearchAgent(
        llm=llm,
        browser_context=webui_manager.bu_browser_context,
        controller=controller
    )
    webui_manager.enhanced_agent = agent # Store reference
    
    # 3. Run
    yield "### Starting Enhanced Agent..."
    
    agent_task = asyncio.create_task(agent.run(goal, url))
    webui_manager.enhanced_agent_task = agent_task

    last_history_len = 0
    while not agent_task.done():
        if agent.history and len(agent.history) > last_history_len:
            log_content = "### Execution History:\n"
            for item in agent.history:
                log_content += f"- **{item['step']}** ({item['status']}): {item['result']}\n"
            yield log_content
            last_history_len = len(agent.history)
        await asyncio.sleep(1)

    history = await agent_task
    
    # 4. Report
    log_content = "## Execution Complete\n"
    for step in history:
        log_content += f"- **{step['step']}** ({step['status']}): {step['result']}\n"
        
    yield log_content

async def stop_enhanced_agent(webui_manager):
    if hasattr(webui_manager, "enhanced_agent") and webui_manager.enhanced_agent:
        await webui_manager.enhanced_agent.stop()
        if hasattr(webui_manager, "enhanced_agent_task") and webui_manager.enhanced_agent_task:
            webui_manager.enhanced_agent_task.cancel()
        return "Stop signal sent. Agent will halt."
    return "No agent running to stop."

def create_enhanced_agent_tab(webui_manager: WebuiManager):
    webui_manager.init_enhanced_agent()
    with gr.Row():
        goal_input = gr.Textbox(label="Goal", value="Complete the quiz", interactive=True)
        url_input = gr.Textbox(label="Starting URL", value="https://www.funtrivia.com/quiz/television/shock-jock-420427.html", interactive=True)
    
    with gr.Row():
        start_btn = gr.Button("Start Enhanced Agent", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")
        
    output_log = gr.TextArea(label="Execution Log", interactive=False, lines=20)
    
    # Bind events
    input_components = get_valid_input_components(list(webui_manager.get_components()))
    
    async def start_wrapper(goal, url, *args):
        comps = dict(zip(input_components, args))
        async for update in safe_execution(run_enhanced_agent, webui_manager, goal, url, comps):
            yield update

    start_btn.click(
        fn=start_wrapper,
        inputs=[goal_input, url_input] + input_components,
        outputs=[output_log]
    )
    
    stop_btn.click(
        fn=lambda: stop_enhanced_agent(webui_manager),
        outputs=[output_log]
    )