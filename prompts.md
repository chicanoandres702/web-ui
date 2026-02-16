# Pro Pilot Prompt Library
# Version: 2.1.0

## 1. Mission: Diagnostic Medic (Auto-Fixer)
**Category:** Debugging
**Context:** Used when the "Problems" tab shows errors or logic is broken.
**Goal:** Fix bugs while maintaining architectural strictness.
> "Analyze the current file. List all active diagnostics. Perform Recursive Debugging by asking 'Why?' three times to find the design flaw, not just the syntax error. Generate a fix. If the fix pushes the file over 100 lines, propose a refactor."

## 2. Mission: Instruction Refresh (Anti-Drift)
**Category:** System
**Context:** Used every 10 messages or when the AI output degrades.
**Goal:** Re-sync the AI with the project's core laws.
> "STOP. Review `AI_CONSTITUTION.md` immediately. Confirm adherence to: 1. The 100-Line Law. 2. The Why Mandate. 3. Functional Programming preference. Acknowledge with 'Systems Online'."

## 3. Mission: Architectural Shrink (Refactoring)
**Category:** Refactoring
**Context:** Used when a file violates the 100-Line Law.
**Goal:** Decompose a monolith into clean components.
> "Analyze [Filename]. Identify distinct responsibilities. Refactor this into: `*.controller.ts` (Orchestration), `*.service.ts` (Business Logic), `*.types.ts` (Interfaces). Ensure strict separation of concerns."

## 4. Mission: Feature Genesis
**Category:** Architecture
**Context:** Creating a new feature from scratch.
**Goal:** Ensure the new feature is modular from day one.
> "I need to implement [Feature Name]. Draft a file structure plan first. Register it in `module.manager.ts`. Create a dedicated directory in `src/features/`. Ensure the Controller is < 100 lines."

## 5. Mission: Debug Orchestrator
**Category:** Debugging
**Context:** User requests debug setup or deep analysis.
**Goal:** Configure environment and set strategic breakpoints.
> "Analyze [Filename]. 1. **Config**: Verify `.vscode/launch.json` exists. 2. **Breakpoint Strategy**: Identify all public interface boundaries and state mutations. Insert 'Logpoints' at entry/exit of Controllers."

## 6. Mission: Context Cartographer (Outline & Tag)
**Category:** Documentation
**Context:** User needs to understand file structure or navigate quickly.
**Goal:** Generate bookmarks, tags, and structural outlines.
> "Analyze [Filename]. 1. **Auto-Tag**: Add comments for navigation: `// @CORE`, `// @API`, `// @STATE`. 2. **Outline**: Generate a Markdown summary of the file's class hierarchy and dependencies."

## 7. Mission: Unit Test Generator
**Category:** Testing
**Context:** Creating tests for a file.
**Goal:** Ensure high code coverage and robustness.
> "Analyze the active file. Generate a comprehensive unit test suite using the project's testing framework. Cover happy paths, edge cases, and error handling. Mock external dependencies where appropriate."

## 8. Mission: Security Auditor
**Category:** Security
**Context:** Reviewing code for vulnerabilities.
**Goal:** Identify and fix security risks.
> "Scan the active file for security vulnerabilities (OWASP Top 10). Check for injection risks, hardcoded secrets, unsafe data handling, and weak authentication. Propose specific fixes."

## 9. Mission: Documentation Scribe
**Category:** Documentation
**Context:** Adding comments and documentation.
**Goal:** Improve code readability and maintainability.
> "Generate JSDoc/DocString documentation for all public interfaces and methods in the active file. Include @param, @return, and usage examples. Ensure tone is professional and clear."

## 10. Mission: Code Reviewer
**Category:** Quality Assurance
**Context:** Reviewing code before commit.
**Goal:** Ensure code quality and adherence to standards.
> "Act as a Senior Engineer. Review the active file for code smells, cognitive complexity, and adherence to SOLID principles. Rate the code 1-10 and list specific improvements."

## 11. Mission: Type Safety Enforcer
**Category:** Refactoring
**Context:** Improving type safety in TypeScript/Python.
**Goal:** Remove 'any' types and ensure strict typing.
> "Analyze the active file for 'any' types or loose typing. Refactor to use strict interfaces, generics, and proper type guards. Ensure strict null checks are respected."

## 12. Mission: Performance Profiler
**Category:** Optimization
**Context:** Optimizing code performance.
**Goal:** Reduce latency and resource usage.
> "Analyze the active file for performance bottlenecks. Look for O(n^2) loops, unnecessary re-renders, memory leaks, or expensive I/O operations. Propose optimized algorithms."

## 13. Mission: API Architect
**Category:** Architecture
**Context:** Designing a new API endpoint.
**Goal:** Create a clean, standard API interface.
> "Design a RESTful/GraphQL API interface for the current feature. Define endpoints, HTTP methods, request/response schemas, and error codes. Ensure standard naming conventions."

## 14. Mission: Git Commit Message
**Category:** Workflow
**Context:** Staging changes for commit.
**Goal:** Generate a semantic commit message.
> "Generate a semantic git commit message based on the staged changes. Use the format 'feat:', 'fix:', 'chore:', 'refactor:'. Include a bulleted list of changes."

## 15. Mission: React Component Generator
**Category:** UI/UX
**Context:** Creating a frontend component.
**Goal:** Create a clean, accessible React component.
> "Create a functional React component for the described requirement. Use Hooks (useState, useEffect), strict TypeScript props, and Tailwind CSS for styling. Ensure accessibility (ARIA)."

## 16. Mission: Feature Modularizer
**Category:** Architecture
**Context:** Refactoring monolithic or legacy code into the Feature Architecture.
**Goal:** Modularize by feature for streamline and reusability.
> "Analyze the active code. Refactor into a modular Feature Component in `src/features/[name]/`. Decompose into: 1. `*.controller.ts` (Orchestration). 2. `*.service.ts` (Business Logic). 3. `*.types.ts` (Interfaces). Ensure strict separation of concerns and adherence to the 100-Line Law."