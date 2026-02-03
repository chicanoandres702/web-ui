# Procedural Memory in AI Agents

## What is Procedural Memory?

In the context of AI and Cognitive Architectures, **Procedural Memory** refers to the storage of "how-to" knowledge. Unlike **Semantic Memory** (which stores facts and concepts) or **Episodic Memory** (which stores specific past experiences), Procedural Memory stores skills, habits, and action sequences.

Think of it as the agent's "muscle memory" for web navigation.

## How it works in this Project

This project utilizes a combination of **Mem0** and explicit **Knowledge Base** files to implement procedural memory.

### 1. Capture
When the agent successfully completes a task or navigates a complex site structure, it can record the sequence of actions that led to success.
*   **Implicit**: The agent's history is analyzed to find successful patterns (via Mem0).
*   **Explicit**: The agent uses tools like `save_site_knowledge` to explicitly save a "recipe" for a specific site (e.g., "To login, click X then Y").

### 2. Storage
This knowledge is stored in the Knowledge Base directory (default: `./tmp/memory`).
*   **Format**: Markdown files named `site_knowledge_[domain].md`.
*   **Content**: Textual descriptions of workflows, selectors, and quirks.

### 3. Retrieval
When the agent visits a domain:
1.  It can use `get_site_knowledge` to check if instructions exist for the current site.
2.  It retrieves the relevant "procedure" to bypass trial-and-error.

## Benefits

*   **Efficiency**: Reduces the number of steps and LLM calls required for repetitive tasks.
*   **Reliability**: Helps the agent handle non-standard UI patterns (e.g., custom dropdowns, anti-bot overlays) by remembering the specific workaround that worked previously.
*   **Adaptability**: The agent builds a personalized playbook for the websites you frequent.

## Related Components
*   `src/controller/custom_controller.py`: Implements `save_site_knowledge` and `get_site_knowledge`.
*   `src/utils/prompts.py`: Instructions (`LEARNING_INSTRUCTIONS`) guiding the agent on when to save and retrieve this knowledge.