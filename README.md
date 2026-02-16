# 🚀 AI Pilot: Your Autonomous IDE Companion

AI Pilot is a VS Code extension that transforms your editor into an intelligent, autonomous development environment. Powered by generative AI, it can understand your project, fix problems, generate code, and execute multi-step development plans.

![AI Pilot UI](https://i.imgur.com/example.png)
  
## Core Features

*   **🧠 Diagnostic Medic:**
    *   **Automated Problem Fixing:** Scans for errors and warnings in your code and provides AI-powered fixes.
    *   **Recursive Debugging:** If a fix fails, the AI analyzes the error and retries, learning from its mistakes.
    *   **Test Generation:** Automatically generates unit tests alongside every fix to ensure correctness.

*   **🤖 Autonomous Task Execution:**
    *   **Project Planner:** Generate a complete project outline from a high-level idea.
    *   **Interactive Planning:** Approve or deny features in the generated outline directly within the editor.
    *   **Task Executor:** The AI breaks down approved features into a step-by-step task list.
    *   **"Go" Button:** Run the autonomous agent to execute the entire task list, from creating files to writing code.

*   **✨ Streamlined UI & UX:**
    *   **Dedicated UI:** A dedicated "AI Pilot" view in the Activity Bar provides a central hub for all features.
    *   **Interactive Chat:** A chat interface for issuing commands and interacting with the AI.
    *   **Task Manager:** A UI to view and manage the agent's task list and execution status.
    *   **Status Bar Integration:** See the AI's current status at a glance.
    *   **Situational Hints:** Get proactive suggestions from the AI as you work.

*   **📚 Comprehensive Knowledge Base:**
    *   **AI Constitution:** A cetral set of rules and principles that govern the AI's behavior.
    *   **LLM-Specific Guides:** Detailed prompting manuals for Gemini, Claude, and GPT to optimize AI performance.
    *   **Automatic Syncing:** The knowledge base and rule files are kept in sync automatically.

*   **🔧 Advanced Configuration:**
    *   **Settings File:** Use the `.vscode/aipilot.json` file to customize prompts and other advanced settings.
    *   **External Auth:** Support for signing in with your own Google account to use its AI services.

## Getting Started

1.  **Install the extension.**
2.  Open the **AI Pilot** view from the Activity Bar.
3.  Run the **"AI Pilot: Initialize Settings File"** command to create `aipilot.json`.
4.  Run the **"AI Pilot: Setup Rules"** command to generate the knowledge base.
5.  Start exploring! Try the **"Generate Project Outline"** command to see the agent in action.

## Changelog

- **v5.1.0 (Current):**
    - **Architectural Refactor:** Full migration to Feature Architecture.
    - **New Features:** Smart Paste, Snippets Manager, Prompt Library.
    - **Improvements:** Enhanced Medic Diagnostic Scan.

- **v1.0.0:**
    - **New Feature:** Autonomous Agent for multi-step task execution.
    - **New Feature:** Interactive Chat UI in the main sidebar.
    - **New Feature:** Centralized prompt configuration via `aipilot.json`.
    - **New Feature:** "Sign in with Google" authentication framework.
    - **New Feature:** Living README feature to auto-document changes.
    - **New Feature:** Situational Hints to provide contextual help.
    - **New Feature:** Task Manager UI to visualize and run agent tasks.
    - **New Feature:** Project Planner to generate and manage project outlines.
    - **New Feature:** Greatly expanded Knowledge Base with guides for multiple LLMs.
    - **New Feature:** UI enhancements including a dedicated Activity Bar icon and Status Bar item.
    - **Enhancement:** Diagnostic Medic now supports test generation and more robust recursive debugging.
    - **Enhancement:** Neural Tab Orchestrator now supports "Focus Mode" and "Dependency Mapping".
    - **Initial Release:** Core features including Diagnostic Medic, Tab Orchestrator, and Global Rule Propagator.