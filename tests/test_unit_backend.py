import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import asyncio
from app.backend import create_llm, get_settings, TaskOrchestrator
from app.models import AgentTaskRequest, TaskStatus, UserFeedback, SubTask


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """
    Fixture to clear the LRU cache of get_settings before each test.
    """
    get_settings.cache_clear()

@pytest.fixture
def orchestrator():
    """
    Fixture to provide a clean TaskOrchestrator instance for each test.
    This ensures that tests are isolated.
    """
    # Since TaskOrchestrator is a singleton, we need to reset its instance
    TaskOrchestrator._instance = None
    return TaskOrchestrator.get_instance()


@patch('app.backend.ChatOllama')
@patch('app.backend.ChatGoogleGenerativeAI')
def test_create_llm_ollama(mock_chat_google, mock_chat_ollama):
    """
    Test that create_llm returns a ChatOllama instance when provider is ollama.
    """
    settings = get_settings()
    settings.LLM_PROVIDER = "ollama"
    settings.MODEL_NAME = "llama3"
    
    llm = create_llm()
    
    mock_chat_ollama.assert_called_once_with(base_url=settings.OLLAMA_BASE_URL, model="llama3")
    assert isinstance(llm, MagicMock)
    mock_chat_google.assert_not_called()

@patch('app.backend.os.path.exists', return_value=False)
@patch('app.backend.ChatOllama')
@patch('app.backend.ChatGoogleGenerativeAI')
def test_create_llm_gemini_api_key(mock_chat_google, mock_chat_ollama, mock_exists):
    """
    Test that create_llm returns a ChatGoogleGenerativeAI instance using an API key.
    """
    settings = get_settings()
    settings.LLM_PROVIDER = "gemini"
    settings.GEMINI_API_KEY = "test-api-key"
    
    llm = create_llm()
    
    mock_chat_google.assert_called_once_with(
        model="gemini-flash-latest",
        google_api_key="test-api-key",
        temperature=0.1
    )
    assert isinstance(llm, MagicMock)
    mock_chat_ollama.assert_not_called()

@patch('app.backend.google.auth.default', return_value=(MagicMock(), "test-project"))
@patch('app.backend.os.path.exists', return_value=True)
@patch('app.backend.ChatOllama')
@patch('app.backend.ChatGoogleGenerativeAI')
def test_create_llm_gemini_service_account(mock_chat_google, mock_chat_ollama, mock_exists, mock_google_auth):
    """
    Test that create_llm returns a ChatGoogleGenerativeAI instance using a service account.
    """
    settings = get_settings()
    settings.LLM_PROVIDER = "gemini"
    
    llm = create_llm()
    
    mock_google_auth.assert_called_once()
    mock_chat_google.assert_called_once()
    assert isinstance(llm, MagicMock)
    mock_chat_ollama.assert_not_called()

@patch('app.backend.ChatOllama')
@patch('app.backend.ChatGoogleGenerativeAI')
def test_create_llm_gemini_model_override(mock_chat_google, mock_chat_ollama):
    """
    Test that create_llm uses the model_override parameter.
    """
    settings = get_settings()
    settings.LLM_PROVIDER = "ollama" # Set a default provider
    
    create_llm(model_override="gemini-pro")
    
    mock_chat_google.assert_called_once_with(
        model="gemini-pro",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.1
    )
    mock_chat_ollama.assert_not_called()


@pytest.mark.asyncio
async def test_add_to_queue(orchestrator: TaskOrchestrator):
    """
    Test adding a task to the orchestrator's queue.
    """
    request = AgentTaskRequest(task="test task")
    task_id = orchestrator.add_to_queue(request)
    
    assert task_id in orchestrator.tasks
    assert orchestrator.tasks[task_id].task_input == "test task"
    assert orchestrator.tasks[task_id].status == TaskStatus.QUEUED
    
    # Check if the task is in the asyncio queue
    queued_item = await orchestrator.queue.get()
    assert queued_item[0] == task_id
    assert queued_item[1] == request


@patch('app.backend.BrowserAgentWrapper')
@pytest.mark.asyncio
async def test_process_lifecycle_approved(mock_agent_wrapper, orchestrator: TaskOrchestrator):
    """
    Test the full lifecycle of a task that is approved by the user.
    """
    # Mock the BrowserAgentWrapper methods
    mock_agent_instance = MagicMock()
    mock_agent_instance.decompose_task = AsyncMock(return_value=[SubTask(description="step 1")])
    mock_agent_instance.run_step = AsyncMock(return_value=True)
    mock_agent_instance.start_session = AsyncMock()
    mock_agent_instance.close_session = AsyncMock()
    mock_agent_wrapper.return_value = mock_agent_instance
    
    # Mock the global callback
    orchestrator.set_callback(AsyncMock())
    
    request = AgentTaskRequest(task="test task", require_confirmation=True)
    task_id = orchestrator.add_to_queue(request)
    
    # Process the task from the queue
    task_info = await orchestrator.queue.get()
    
    # Run the lifecycle in a separate task to simulate the worker
    lifecycle_task = asyncio.create_task(orchestrator._process_lifecycle(*task_info))
    
    # Wait for the task to reach the waiting_for_user state
    await asyncio.sleep(0.1)
    assert orchestrator.tasks[task_id].status == TaskStatus.WAITING_FOR_USER
    
    # Simulate user approval
    feedback = UserFeedback(task_id=task_id, approved=True)
    orchestrator.handle_feedback(feedback)
    
    # Wait for the lifecycle to complete
    await lifecycle_task
    
    # Assertions
    mock_agent_instance.decompose_task.assert_called_once()
    mock_agent_instance.start_session.assert_called_once()
    mock_agent_instance.run_step.assert_called_once()
    mock_agent_instance.close_session.assert_called_once()
    
    final_task_state = orchestrator.tasks[task_id]
    assert final_task_state.status == TaskStatus.COMPLETED
    assert final_task_state.plan[0].status == "completed"

@patch('app.backend.BrowserAgentWrapper')
@pytest.mark.asyncio
async def test_process_lifecycle_cancelled(mock_agent_wrapper, orchestrator: TaskOrchestrator):
    """
    Test the lifecycle of a task that is cancelled by the user.
    """
    mock_agent_instance = MagicMock()
    mock_agent_instance.decompose_task = AsyncMock(return_value=[SubTask(description="step 1")])
    mock_agent_wrapper.return_value = mock_agent_instance
    
    orchestrator.set_callback(AsyncMock())
    
    request = AgentTaskRequest(task="test task", require_confirmation=True)
    task_id = orchestrator.add_to_queue(request)
    
    task_info = await orchestrator.queue.get()
    lifecycle_task = asyncio.create_task(orchestrator._process_lifecycle(*task_info))
    
    await asyncio.sleep(0.1)
    assert orchestrator.tasks[task_id].status == TaskStatus.WAITING_FOR_USER
    
    # Simulate user cancellation
    feedback = UserFeedback(task_id=task_id, approved=False)
    orchestrator.handle_feedback(feedback)
    
    await lifecycle_task
    
    # The task should be stopped and run_step should not be called
    mock_agent_instance.decompose_task.assert_called_once()
    mock_agent_instance.run_step.assert_not_called()
    
    final_task_state = orchestrator.tasks[task_id]
    assert final_task_state.status == TaskStatus.STOPPED
