# Things to Avoid in AI Pilot Development

This document outlines anti-patterns and practices to avoid when developing the AI Pilot project.  Following these guidelines will help maintain code quality, prevent bugs, and ensure a consistent development experience.

## General

*   **Over-Engineering:**  Avoid implementing overly complex solutions when simpler approaches are sufficient.  Keep the code as straightforward as possible.
*   **Premature Optimization:**  Don't optimize code before identifying actual performance bottlenecks. Focus on writing clear and functional code first.
*   **Ignoring Errors:**  Always handle errors gracefully.  Don't suppress or ignore exceptions without proper logging and handling.

## VS Code Extension Specific

*   **Blocking the UI Thread:**  Avoid long-running synchronous operations in the main UI thread. Use asynchronous operations to prevent freezing the editor.
*   **Leaking Disposables:**  Ensure that all disposable resources (e.g., event listeners, subscriptions) are properly disposed of to prevent memory leaks.
*   **Hardcoding Configuration:**  Avoid hardcoding configuration values. Use the VS Code configuration API to allow users to customize settings.

## Architecture

*   **Monolithic Features:** Avoid placing all feature logic in a single file. Adhere to the Feature Architecture: separate Orchestration (Controller) from Business Logic (Service).
*   **Circular Dependencies:** Be mindful of imports between features. Use the `FeatureLoader` for dependency injection to manage lifecycle and dependencies cleanly.