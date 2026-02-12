import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from app.browser import start_chrome_with_debug_port, BrowserAgentWrapper
from app.models import SubTask

@pytest.mark.asyncio
@patch('app.browser.asyncio.create_subprocess_exec')
@patch('app.browser.aiohttp.ClientSession')
@patch('app.browser.os.path.exists', return_value=True)
@patch('app.browser.sys.platform', 'win32')
async def test_start_chrome_win32(mock_platform, mock_exists, mock_session, mock_exec):
    """
    Test start_chrome_with_debug_port on Windows.
    """
    # Mock the subprocess
    mock_proc = AsyncMock()
    mock_proc.wait = AsyncMock()
    mock_proc.returncode = 0
    mock_exec.return_value = mock_proc

    # Mock the CDP check
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_resp
    
    process, user_data_dir = await start_chrome_with_debug_port()
    
    assert process is not None
    assert user_data_dir is not None
    mock_exec.assert_called()
    
    # Check that it tried a windows path first
    first_call_args = mock_exec.call_args_list[0].args
    assert 'msedge.exe' in first_call_args[0]


@pytest.mark.asyncio
@patch('app.browser.asyncio.create_subprocess_exec', new_callable=AsyncMock)
async def test_start_chrome_browser_not_found(mock_exec):
    """
    Test that start_chrome_with_debug_port raises an error if no browser is found.
    """
    mock_exec.side_effect = FileNotFoundError
    
    with pytest.raises(RuntimeError, match="Could not find a compatible browser"):
        await start_chrome_with_debug_port()


@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
@patch('app.browser.Browser')
@patch('app.browser.start_chrome_with_debug_port')
def browser_agent_wrapper(mock_start_chrome, mock_browser, mock_llm):
    """
    Fixture for BrowserAgentWrapper with mocked dependencies.
    """
    # Configure mocks
    mock_start_chrome.return_value = (AsyncMock(), "test_dir")
    
    mock_browser_instance = MagicMock()
    mock_browser_instance.new_session = AsyncMock()
    mock_browser.return_value = mock_browser_instance
    
    # Return the wrapper instance
    return BrowserAgentWrapper(llm=mock_llm)

@pytest.mark.asyncio
async def test_browser_agent_start_close_session(browser_agent_wrapper: BrowserAgentWrapper):
    """
    Test the start_session and close_session methods of BrowserAgentWrapper.
    """
    await browser_agent_wrapper.start_session()
    
    # Assertions for start_session
    assert browser_agent_wrapper.chrome_process is not None
    assert browser_agent_wrapper.browser is not None
    assert browser_agent_wrapper.session is not None
    browser_agent_wrapper.browser.new_session.assert_called_once()
    
    # Now test close_session
    await browser_agent_wrapper.close_session()
    
    assert browser_agent_wrapper.chrome_process is None
    assert browser_agent_wrapper.browser is None
    assert browser_agent_wrapper.session is None


@pytest.mark.asyncio
@patch('app.browser.BUAgent')
async def test_browser_agent_run_step(mock_bu_agent, browser_agent_wrapper: BrowserAgentWrapper):
    """
    Test the run_step method of BrowserAgentWrapper.
    """
    # Mock the BUAgent from browser-use
    mock_bu_agent_instance = MagicMock()
    mock_bu_agent_instance.run = AsyncMock(return_value=True)
    mock_bu_agent.return_value = mock_bu_agent_instance
    
    # Ensure session is started
    await browser_agent_wrapper.start_session()
    
    # Create a mock for the callback
    mock_callback = AsyncMock()

    # Run the step
    success = await browser_agent_wrapper.run_step(
        description="test step",
        class_name="test class",
        callback=mock_callback
    )

    assert success is True
    mock_bu_agent.assert_called_once()
    mock_bu_agent_instance.run.assert_called_once()
