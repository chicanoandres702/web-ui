"""
This module contains the configuration and dependency file definitions for the project setup.
"""
CONFIG_FILES = {
    "requirements.txt": '''fastapi
uvicorn[standard]
pydantic>=2.0
pydantic-settings
langchain
langchain-core
langchain-google-genai
langchain-openai
langchain-community
langchain-ollama
huggingface_hub
browser-use
tqdm
requests
python-multipart
python-dotenv
google-auth
google-auth-oauthlib
playwright
jinja2
itsdangerous
aiohttp
pytest-asyncio
'''
}
