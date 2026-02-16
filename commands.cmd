@echo off
title AI Pilot: Gemini Grounding Mission (Ultimate Knowledge Base Edition)
setlocal

set ERROR_OCCURRED=0

echo 🚀 Initializing AI Pilot: The Ultimate Knowledge Base Sequence...
echo --------------------------------------------

:: 1. Create Directory Structure
echo Creating robust folder structure...
if not exist ".google" mkdir .google
if not exist ".github" mkdir .github
if not exist ".ai-pilot" mkdir .ai-pilot
if not exist ".ai-pilot\knowledge-vault" mkdir .ai-pilot\knowledge-vault

:: Backup existing files
if exist "AI_CONSTITUTION.md" copy "AI_CONSTITUTION.md" "AI_CONSTITUTION.md.bak"
if exist ".ai-pilot\knowledge-vault\FEATURE_MANIFEST.md" copy ".ai-pilot\knowledge-vault\FEATURE_MANIFEST.md" ".ai-pilot\knowledge-vault\FEATURE_MANIFEST.md.bak"
if exist ".ai-pilot\knowledge-vault\MODULAR_STANDARDS.md" copy ".ai-pilot\knowledge-vault\MODULAR_STANDARDS.md" ".ai-pilot\knowledge-vault\MODULAR_STANDARDS.md.bak"
if exist ".google\config.yaml" copy ".google\config.yaml" ".google\config.yaml.bak"
if exist ".ai-pilot\knowledge-vault\gemini_logic.md" copy ".ai-pilot\knowledge-vault\gemini_logic.md" ".ai-pilot\knowledge-vault\gemini_logic.md.bak"

:: 2. Generate AI_CONSTITUTION.md (The Massive Knowledge Base)
echo Generating Master Constitution Knowledge Base...
echo (
echo # AI_CONSTITUTION.md
echo # Version: 5.1.0 - Comprehensive Knowledge Base Edition
echo.
echo ## Section 1: Core Architectural Directives
echo 1. The 100-Line Law: Files MUST NOT exceed 100 lines. Smaller files keep AI Token Density high and focus sharp.
echo 2. The Why Mandate: Comments must explain the design intent, not the syntax.
echo 3. Modular First: Prioritize small, reusable components. Use Interfaces to define clear boundaries.
echo 4. Clean Code: No magic numbers. Use named constants. Prioritize readability over cleverness.
echo 5. Contextual Naming: Files must use descriptive prefixes (e.g., 'setup.service.ts') to prevent AI context loss.
echo 6. Feature Architecture: All business logic must reside in 'src/features/^<feature^>/'.
echo.
echo ## Section 2: Advanced Logic Archetypes
echo ### The Diagnostic Medic (Mission: Medic)
echo - Before any fix, perform Recursive Debugging. Identify the design flaw, not just the syntax error.
echo - Root Cause Analysis (RCA) is required for every bug report.
echo - Diagnostic Listener: Monitor the Problems tab. Categorize errors before proposing code.
echo.
echo ### Anti-Drift Protocol
echo - AI Instruction Drift occurs every 10-15 messages.
echo - Perform a Pilot Refresh to re-ground the LLM in these rules periodically.
echo.
echo ### Neural Tab Orchestration
echo - Keep Neighbor Files open to give the AI cross-file context (e.g., Logic + Interface).
echo - Focus Mode: Close tabs unused for 15+ minutes to reduce Context Poisoning.
echo.
echo ## Section 3: Multi-Model Intelligence (The LLM Encyclopedia)
echo - Google Gemini: Master of massive context (1M+ tokens). Use for deep debugging and full-repo analysis.
echo - Claude (Anthropic): Master of structural integrity. Use for complex logic and XML-based instructions.
echo - GPT (OpenAI): Master of rapid prototyping. Use for UI/UX logic and creative data manipulation.
echo.
echo ## Section 4: The Token Economy ^& Performance
echo - AI intelligence is inversely proportional to context noise.  REM "&" can cause problems; avoid in batch scripts or escape it.
echo - Delete unused code, close unused tabs, and keep files modular to ensure the AI stays Expert Level.
echo - Predictive Logic: Always anticipate the next 3 steps in the development lifecycle.
echo.
echo ## Section 5: Language-Specific Standards
echo - TypeScript/JS: Use Functional Programming. Avoid 'any'. Prefer 'unknown' for unsafe types.
echo - Python: Follow PEP8. Use Type Hints for every function. Use 'pydantic' for data validation.
echo - C#: Use 'internal sealed' for logic classes. Adhere to SOLID principles strictly.
echo.
echo ## Section 6: Educational and Professional Ethics
echo - Student Friendly: Maintain documentation that explains complex concepts simply.
echo - Safety First: No harmful, illegal, or age-inappropriate content. 
) > AI_CONSTITUTION.md
:: 2.1. Generate FEATURE_MANIFEST.md (Modular Feature Registry)
echo Generating Feature Manifest...
(
echo # FEATURE_MANIFEST.md
echo.
echo ## Feature Directory Structure
echo - All features must reside in: `src/features/^<feature-name^>/`
echo - Each feature folder must contain:
echo   - `components/`: UI elements
echo   - `services/`: Business logic and API calls
echo   - `hooks/` or `utils/`: Logic helpers
echo   - `types/`: Interface definitions
echo.
echo ## Registration
echo - New features must be registered here to maintain AI context of the system architecture.
) > .ai-pilot\knowledge-vault\FEATURE_MANIFEST.md

:: 2.2. Generate MODULAR_STANDARDS.md (Reusability Guidelines)
echo Generating Modular Standards...
(
echo # MODULAR_STANDARDS.md
echo.
echo ## Component Reusability
echo 1. Atomic Design: Break UI into Atoms, Molecules, and Organisms.
echo 2. Pure Functions: Logic services should be side-effect free where possible.
echo 3. Dependency Injection: Pass dependencies rather than hard-coding imports to facilitate testing.
echo 4. Single Responsibility: One file, one export, one purpose.
) > .ai-pilot\knowledge-vault\MODULAR_STANDARDS.md
:: 3. Sync to all AI manifest files
echo Syncing manifests (Cursor, Claude, Copilot, Windsurf)...
copy /y AI_CONSTITUTION.md .cursorrules >nul
copy /y AI_CONSTITUTION.md CLAUDE.md >nul
copy /y AI_CONSTITUTION.md .windsurfrules >nul
copy /y AI_CONSTITUTION.md .github\copilot-instructions.md >nul

:: 4. Generate Gemini Config (.google/config.yaml)
echo Optimizing Gemini Code Assist...
(
echo # Gemini Code Assist Project Context
echo project_context:
echo   rules_file: "AI_CONSTITUTION.md"
echo   documentation_path: ".ai-pilot/knowledge-vault"
echo   coding_standards:
echo     - "Strict 100-line limit per file"
echo     - "Recursive Debugging (Medic Mode)"
echo     - "Design Intent documentation"
echo     - "Functional Programming Pref"
) > .google\config.yaml

:: 5. Generate Advanced Gemini Logic Vault
echo Finalizing Gemini Logic Vault...
(
echo # Gemini Omni-Model Logic and Commands
echo # Version: 11.0.0
echo.
echo ## Slash Commands (Copy/Paste to Gemini^)
echo - /medic:scan -^> Perform a Diagnostic Scan for Constitution violations and logic bugs.
echo - /medic:fix -^> Apply a modular fix that strictly respects the 100-line limit.
echo - /arch:shrink -^> Analyze a file and propose a refactor into smaller, cleaner modules.
echo - /pilot:refresh -^> Force the AI to re-read the AI_CONSTITUTION.md and sync project state.
echo - /pilot:docs -^> Generate student-friendly documentation for the current module.
echo.
echo ## Reasoning Step-by-Step (The Thinking Brain^)
echo 1. [THOUGHT]: Determine the root cause and check against the Constitution.
echo 2. [LOGIC]: Draft a modular plan that minimizes code changes while maximizing clarity.
echo 3. [CODE]: Implement with "Why" comments. Ensure no file grows beyond 100 lines.
echo 4. [VERIFY]: Test logic against the "100-Line Law" and "The Why Mandate."
echo.
echo ## Deep Context Patterns
echo - [RAG]: When analyzing, reference the '.ai-pilot/knowledge-vault' for platform-specific quirks.
echo - [CHAIN]: Link multiple logical steps together before outputting final code.
echo - [NAMING]: Use descriptive filenames (e.g., 'feature.component.ts') to anchor AI context.
echo - [STRUCTURE]: Locate logic in 'src/features/{feature}/'.
) > .ai-pilot\knowledge-vault\gemini_logic.md

echo --------------------------------------------

goto :end

:error_rollback
echo An error occurred. Rolling back changes...
if exist "AI_CONSTITUTION.md.bak" copy /y "AI_CONSTITUTION.md.bak" "AI_CONSTITUTION.md" >nul
if exist ".ai-pilot\knowledge-vault\FEATURE_MANIFEST.md.bak" copy /y ".ai-pilot\knowledge-vault\FEATURE_MANIFEST.md.bak" ".ai-pilot\knowledge-vault\FEATURE_MANIFEST.md" >nul
if exist ".ai-pilot\knowledge-vault\MODULAR_STANDARDS.md.bak" copy /y ".ai-pilot\knowledge-vault\MODULAR_STANDARDS.md.bak" ".ai-pilot\knowledge-vault\MODULAR_STANDARDS.md" >nul
if exist ".google\config.yaml.bak" copy /y ".google\config.yaml.bak" ".google\config.yaml" >nul
if exist ".ai-pilot\knowledge-vault\gemini_logic.md.bak" copy /y ".ai-pilot\knowledge-vault\gemini_logic.md.bak" ".ai-pilot\knowledge-vault\gemini_logic.md" >nul

set ERROR_OCCURRED=1

:end

if %ERROR_OCCURRED% equ 1 (
  echo Operation failed. Changes have been rolled back.
) else (
echo ✨ Mission Complete!
echo Your project is now grounded with a Professional AI Knowledge Base.
)

pause
exit /b %ERROR_OCCURRED%