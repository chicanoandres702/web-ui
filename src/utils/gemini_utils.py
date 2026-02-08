import os
from google.generativeai import GenerativeModel

class GeminiConfig:
    api_key: str = os.getenv("GOOGLE_API_KEY")


async def get_gemini_models():
    """
    Retrieves available Gemini models.

    Returns:
        list: A list of Gemini model names.
    """
    try:
        # Initialize the GenerativeModel with a default model name.
        # The API key is implicitly obtained from the GOOGLE_API_KEY environment variable.
        model = GenerativeModel("gemini-1.5-pro-latest")
        
        # List available models.
        available_models = [model.name for model in model.list_models()]
        return available_models
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")
        
        return []


get_gemini_models_wrapper = get_gemini_models