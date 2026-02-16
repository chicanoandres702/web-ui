# AI Pilot: Architecture & System Design

## 📂 Documentation & Resources

### 1. Project Documentation (docs/)
- **File**: `ARCHITECTURE.md` (This file) - System outline.
- **File**: `ANTI_PATTERNS.md` - Development guardrails.
- **File**: `CHANGELOG.md` - Version history.
- **File**: `README.md` - User guide.

### 2. AI Knowledge Base (resources/ai/)
- **File**: `AI_CONSTITUTION.md` - Core rules.
- **File**: `CLAUDE.md` - Anthropic specific guides.
- **File**: `GEMINI.md` - Google specific guides.

## 🧩 Core System

### 1. `ModuleManager` (src/core/module.manager.ts)
- **Role**: Composition Root / State Manager
- **Dependencies**: `FeatureLoader`
- **@CORE**: `registerFeatures()` - Bootstraps the extension.
- **@CORE**: `toggle()` - Global enable/disable switch.

### 2. `FeatureLoader` (src/features/feature.loader.ts)
- **Role**: Dependency Injection Container / Registry
- **Dependencies**: All Feature Controllers/Factories
- **@CORE**: `loadFeatures()` - Instantiates and returns all feature disposables.
- **@CORE**: `loadMedic()` - Wires up Diagnostic Medic.

### 3. `CoreLoader` (src/core/core.loader.ts)
- **Role**: Core Services Initializer
- **@CORE**: `loadCore()` - Bootstraps essential services.

### 4. `ToolLoader` (src/features/tool.loader.ts)
- **Role**: External Tools Registry
- **@CORE**: `loadTools()` - Integrates external capabilities.

## 📦 Module: Setup Generator

### 1. `SetupFactory` (src/features/setup/setup.factory.ts)
- **Role**: Dependency Injection
- **@CORE**: `getTelemetry()` - Exposes shared instance.
- **@CORE**: `createController()` - Assembles module.
- **@CORE**: `createWizardController()` - Assembles wizard.

### 2. `SetupController` (src/features/setup/setup.controller.ts)
- **Role**: Orchestrator / Interaction Handler
- **Dependencies**: `SetupService`, `SetupTelemetry`
- **@API**: `generateScripts()` - Triggers generation flow.

### 3. `SetupService` (src/features/setup/setup.service.ts)
- **Role**: Business Logic / File I/O
- **Dependencies**: `SetupTemplates`, `SetupTelemetry`, `SetupValidator`
- **@API**: `generateScripts(rootPath)` - Writes files to disk.
- **@CORE**: `writeFile(root, name, content)` - Low-level write.

### 4. `SetupTemplates` (src/features/setup/setup.templates.ts)
- **Role**: Data / Templates
- **Dependencies**: `SetupConfig`
- **@DATA**: `getInstallPy()` - Python installer script.
- **@DATA**: `getSetupCmd()` - Windows batch script.
- **@DATA**: `getSetupSh()` - Unix shell script.

### 5. `SetupValidator` (src/features/setup/setup.validator.ts)
- **Role**: Integrity Check
- **@CORE**: `validate(rootPath)` - Checks file existence/content.

### 6. `SetupTelemetry` (src/features/setup/setup.telemetry.ts)
- **Role**: Observability
- **@CORE**: `log(message)` - Writes to Output Channel.

### 7. `SetupConfig` (src/features/setup/setup.config.ts)
- **Role**: Configuration
- **@API**: `getPythonCommand()` - Reads VS Code settings.

### 8. `SetupWizardController` (src/features/setup/setup.wizard.controller.ts)
- **Role**: UI Orchestrator
- **Dependencies**: `SetupWizardTelemetry`
- **@API**: `showWizard()` - Opens webview.

### 9. `SetupWizardView` (src/features/setup/setup.wizard.view.ts)
- **Role**: UI Presentation
- **@CORE**: `getWebviewContent()` - Returns HTML.

### 10. `SetupWizardTelemetry` (src/features/setup/setup.wizard.telemetry.ts)
- **Role**: Wizard Analytics
- **@CORE**: `logOpen/Generate/Close()` - Tracks usage.

## 📦 Module: Snippets Manager

### 1. `SnippetsController` (src/features/snippets/snippets.controller.ts)
- **Role**: Orchestrator
- **Dependencies**: `SnippetsService`
- **@API**: `captureSnippet()`, `insertSnippet()`

### 2. `SnippetsService` (src/features/snippets/snippets.service.ts)
- **Role**: Logic / Storage
- **@CORE**: `saveSnippet()`, `getSnippets()`

## 📦 Module: Smart Paste

### 1. `SmartPasteController` (src/features/smartpaste/smartpaste.controller.ts)
- **Role**: Event Listener / Orchestrator
- **Dependencies**: `SmartPasteService`
- **@API**: `handlePaste()` - Intercepts paste event.

### 2. `SmartPasteService` (src/features/smartpaste/smartpaste.service.ts)
- **Role**: Business Logic (Sanitization)
- **@CORE**: `sanitize(text)` - Removes smart quotes/zero-width spaces.

## 📦 Module: Prompt Library

### 1. `PromptController` (src/features/prompt/prompt.controller.ts)
- **Role**: Orchestrator
- **Dependencies**: `PromptService`
- **@API**: `insertPrompt()` - Shows quick pick and inserts.

### 2. `PromptService` (src/features/prompt/prompt.service.ts)
- **Role**: Data Access
- **@CORE**: `getPrompts()` - Reads/Parses prompts.md.

## 📦 Module: Profiles

### 1. `ProfilesController` (src/features/profiles/profiles.controller.ts)
- **Role**: Orchestrator
- **Dependencies**: `ProfilesService`
- **@API**: `switchProfile()` - User command.

### 2. `ProfilesService` (src/features/profiles/profiles.service.ts)
- **Role**: State Management
- **Dependencies**: `ProfilesConfig`
- **@CORE**: `setActiveProfile()` - Updates state.
- **@CORE**: `detectProfile()` - Auto-detection logic.

### 3. `ProfilesConfig` (src/features/profiles/profiles.config.ts)
- **Role**: Schema / Validation
- **@CORE**: `validateSchema()` - Ensures profile integrity.

## 📦 Module: Workflow Generator

### 1. `WorkflowController` (src/features/workflow/workflow.controller.ts)
- **Role**: Orchestrator
- **@API**: `generateWorkflows()` - Creates CI/CD assets.

### 2. `WorkflowService` (src/features/workflow/workflow.service.ts)
- **Role**: File I/O
- **@CORE**: `generateFiles()` - Writes .github and scripts.