# 1. Create the AI infrastructure & Feature Modules
$dirs = ".gemini/commands/arch", ".gemini/commands/refactor", ".gemini/commands/test", "src/core", "src/features/auth", "src/features/agents", "src/features/llm", "src/ui"
foreach ($dir in $dirs) { New-Item -ItemType Directory -Force -Path $dir }

# 2. Initialize Python Packages (Modularity)
$pyModules = "src/core", "src/features/auth", "src/features/agents", "src/features/llm"
foreach ($mod in $pyModules) { if (!(Test-Path "$mod/__init__.py")) { New-Item -ItemType File -Force -Path "$mod/__init__.py" } }

# 3. Build the 'AI_GUIDE.md' (Your Workstation Manual)
Set-Content -Path "AI_GUIDE.md" -Value @"
# 🤖 AI Workstation Guide
Use these tips to keep Gemini's "Brain" sharp:

### 💡 The 'Anchor' Rule
Before starting a task, open the **GEMINI.md** and the **Contract** file. 
Gemini 'sees' what you have open. Pinned tabs = Priority Context.

### 🚀 Available Macros
- `/feature [Name]` -> Create root feature folder.
- `/core [Name]`    -> Add shared logic to src/core.
- `/refactor:shrink` -> Split logic to keep files < 100 lines.
- `/refactor:module` -> Move flat logic into `src/features/[feature]`.
- `/refactor:sync`   -> Force Service to match its Contract.

### 🧠 Workstation Handling
- **Flat Folders**: If you find yourself nesting, stop. Move to root.
- **Factory First**: If you see 'new Service()', tell Gemini: "Refactor to use Factory."
"@

# 4. Create the 'Blueprint' command
Set-Content -Path ".gemini/commands/blueprint.toml" -Value 'description = "Explains how to handle the workstation."`nprompt = "Read AI_GUIDE.md and explain the best way to handle the current file based on our rules."'

Write-Host "✅ AI Extension Layer Installed!" -ForegroundColor Green