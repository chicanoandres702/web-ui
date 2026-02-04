# Model Heuristics & Priority List

This document explains the new Model Priority List feature and how it powers the **Smart Retry** and **Cost Saver** heuristics for the Browser Use Agent.

## The Concept

Instead of having separate, fixed models for "main", "retry", and "cheap" tasks, you now define a single, prioritized pool of available models. The agent intelligently switches between these models based on its real-time performance.

## How It Works

### 1. The Model Priority List

In the **Agent Settings** tab, you'll find a new "Heuristic Model Switching" section with a table:

*   **Priority**: A number indicating the model's strength. **Lower numbers mean higher priority** (e.g., 1 is the strongest, 99 is the weakest). The "Main LLM" is implicitly Priority 0.
*   **Provider, Model Name, etc.**: Standard LLM connection details.

### 2. Smart Retry (Upgrading on Failure)

*   **Trigger**: If the agent fails 2 or more consecutive steps.
*   **Action**: The agent searches the priority list for the next-strongest model (i.e., the one with the highest priority number that is still *lower* than its current model's priority).
*   **Example**: If the agent is using a Priority 10 model and fails, it will look for a model with a priority between 1 and 9. It will pick the one closest to 9 (e.g., Priority 8). If it's on the Main LLM (Priority 0) and fails, it will switch to the best model in the list (e.g., Priority 1).
*   **Reset**: Once it switches, the failure counter resets to give the new model a fair chance.

### 3. Cost Saver (Downgrading on Success)

*   **Trigger**: If the agent successfully completes 3 consecutive steps.
*   **Action**: The agent searches the list for a weaker/cheaper model (i.e., one with a *higher* priority number).
*   **Example**: If the agent is on the Main LLM (Priority 0) and succeeds 3 times, it will look for the model with the lowest priority number in the list (e.g., Priority 10) and switch to it.
*   **Reset**: The success counter resets after switching.

### 4. Fail-Safe

*   If any model (other than the Main LLM) fails even **once**, the agent immediately reverts to the Main LLM (Priority 0) to ensure the task can proceed reliably. It will then need to build up 3 successes again before attempting to downgrade for cost savings.

This system provides a flexible and powerful way to balance cost, speed, and capability, allowing the agent to adapt its "brainpower" to the difficulty of the task at hand.