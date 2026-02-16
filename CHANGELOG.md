# Change Log

All notable changes to the "aipilot" extension will be documented in this file.

Check [Keep a Changelog](http://keepachangelog.com/) for recommendations on how to structure this file.

## [5.1.0] - 2023-10-27

### Architectural Refactor
- **Modular Architecture**: Migrated all business logic from `src/` root and `src/libs/` to domain-specific directories in `src/features/`.
- **Feature Loaders**: Introduced `FeatureLoader`, `CoreLoader`, and `ToolLoader` to decouple feature registration from state management.
- **Naming Convention**: Enforced `feature.component.ts` pattern across the codebase for better AI context retrieval.
- **100-Line Law**: Refactored `ModuleManager` and various controllers to strictly adhere to the 100-line limit per file.

### New Features
- **Snippets Manager**: Added a robust snippet management system (`src/features/snippets/`) with Capture, Insert, Delete, and Copy to Clipboard capabilities.
- **Smart Paste**: Implemented `SmartPaste` feature (`src/features/smartpaste/`) to automatically sanitize text (remove smart quotes, zero-width spaces) on paste.
- **Prompt Library**: Expanded `prompts.md` with 15 professional mission profiles and moved it to `src/features/prompt/`.

### Improvements
- **Medic**: Enhanced diagnostic scan to detect Naming Convention and Feature Architecture violations.
- **Setup**: Updated setup scripts to generate the new `AI_CONSTITUTION.md` v5.1.0 with the "Feature Architecture" directive.
- **Profiles**: Consolidated profile logic into `src/features/profiles/` with schema validation and auto-detection.

### Removed
- Deleted obsolete `src/libs/` directory.
- Removed legacy root-level controllers and services (`sync.controller.ts`, `settings.service.ts`, etc.).