from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_client_id: str = None
    google_client_secret: str = None
    secret_key: str = "your_default_secret_key"
    rate_limit_seconds: int = 60
    max_requests_per_minute: int = 20
    openai_endpoint: str = None
    anthropic_endpoint: str = None
    azure_openai_api_version: str = None
    deepseek_endpoint: str = None
    mistral_endpoint: str = None
    ollama_endpoint: str = None
    alibaba_endpoint: str = None
    modelscope_endpoint: str = None
    moonshot_endpoint: str = None
    unbound_endpoint: str = None
    siliconflow_endpoint: str = None
    ibm_endpoint: str = None
    grok_endpoint: str = None
    default_llm: str = None
    google_application_credentials: str = None
    anonymized_telemetry: str = None
    browser_use_logging_level: str = None
    browser_debugging_port: str = None
    browser_debugging_host: str = None
    keep_browser_open: str = None
    use_own_browser: str = None
    resolution: str = None
    resolution_width: str = None
    resolution_height: str = None
    vnc_password: str = None

    class Config:
        env_file = ".env"

settings = Settings()
