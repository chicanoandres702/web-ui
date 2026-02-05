# Browser Use Web UI

A lightweight, advanced Web UI for the [Browser Use](https://github.com/browser-use/browser-use) agent. This interface allows you to control autonomous browser agents, perform deep research, and interact with LLMs like OpenAI, Google Gemini, and Ollama.

## Features

- **Interactive Web UI**: Real-time browser view and control.
- **Multiple Agent Types**: Standard Browser Agent and Deep Research Agent.
- **Google Login Integration**: Sign in with your Google account to use Gemini models without manually handling API keys.
- **Local LLM Support**: Integration with Ollama.
- **File Management**: Upload files and view generated artifacts.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd web-ui-1
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    playwright install
    ```
    *Note: Ensure you have `langchain-google-genai`, `google-auth-oauthlib`, and `itsdangerous` installed for Google Login support.*

## Configuration

### 1. Environment Variables (`.env`)
Create a `.env` file in the root directory to store your API keys and configuration.

```env
# Security (Required for Session Management)
SECRET_KEY=change_this_to_a_random_string

# Google OAuth (Required for "Sign in with Google")
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Optional: Default API Keys (if not using UI input)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

### 2. Google OAuth Setup (For Gemini)
To enable the "Sign in with Google" button:

1.  Go to the Google Cloud Console.
2.  Create a new project.
3.  Enable the **Google Generative Language API**.
4.  **Configure OAuth Consent Screen**:
    *   Go to **APIs & Services > OAuth consent screen**.
    *   Select **External** and Create.
    *   **Scopes**: Click **Add or Remove Scopes** and manually add `https://www.googleapis.com/auth/cloud-platform` and `https://www.googleapis.com/auth/generative-language.peruserquota`.
    *   **Test Users**: Add your Google email address. This is required for the app to work while in "Testing" status.
5.  Go to **APIs & Services > Credentials**.
6.  Create **OAuth client ID** credentials (Application type: Web application).
7.  Set **Authorized redirect URIs** to: `http://localhost:8000/auth/callback` and `http://127.0.0.1:8000/auth/callback`
8.  Download the JSON file, rename it to `client_secret.json`, and place it in the project root.
    *   *Alternatively, copy the Client ID and Secret into your `.env` file.*

**Note:** Both `.env` and `client_secret.json` are ignored by git to keep your credentials safe.

### 3. Service Account Setup (Optional - No Login Required)
If you prefer not to use the "Sign in with Google" flow, you can use a Service Account:
1.  Go to **IAM & Admin > Service Accounts** in Google Cloud Console.
2.  Create a Service Account and grant it the **Vertex AI User** or **Generative Language User** role.
3.  Create a **JSON Key** for this account.
4.  Rename the downloaded file to `service_account.json` and place it in the project root.
5.  The app will automatically detect and use it if you are not logged in.

## Usage

1.  **Start the Server**:
    ```bash
    python server.py
    ```

2.  **Open the UI**:
    Navigate to `http://localhost:8000` in your browser.

3.  **Select LLM**:
    - Choose **Gemini (Google)** and click "Sign in with Google" to use your account quota.
    - Or select **OpenAI** / **Anthropic** and enter your API Key.
    - Or select **Ollama** for local inference.

## Credits

This project is a Web UI wrapper built around the incredible **Browser Use** library.

- **Core Library**: Browser Use
- **Original Developers**: The Browser Use Team

Please support the original project by starring their repository!