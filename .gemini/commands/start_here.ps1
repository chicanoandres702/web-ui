# 1. Create Folder Structure
$dirs = ".gemini/commands/arch", ".gemini/commands/refactor", ".gemini/commands/test"
foreach ($dir in $dirs) { New-Item -ItemType Directory -Force -Path $dir }

# 2. Create GEMINI.md
Set-Content -Path "GEMINI.md" -Value @"
# Project Rules: AI-Readable Architecture
- **Structure**: Flat root folders per feature. No nesting > 2 levels.
- **Naming**: Use [Feature][Type] (e.g., AuthService.ts, AuthFactory.ts).
- **Instantiation**: ALWAYS use the Factory. Never use 'new' for services.
- **Small Classes**: Keep files under 100 lines. Refactor if they grow.
- **Logic**: Use early returns. Keep it left-aligned.
"@

# 3. Create .aiexclude
Set-Content -Path ".aiexclude" -Value "node_modules/`nbin/`nobj/`ndist/`n.vs/`n.vscode/`n*.log"

# 4. Create Commands
Set-Content -Path ".gemini/commands/feature.toml" -Value 'description = "Quick-start feature."`nprompt = "Create a root folder for {{args}} per GEMINI.md."'
Set-Content -Path ".gemini/commands/arch/new.toml" -Value 'description = "Deep blueprint."`nprompt = "Create root folder for {{args}}. Generate Contract and Factory first."'
Set-Content -Path ".gemini/commands/refactor.toml" -Value 'description = "Quick clean."`nprompt = "Fix naming and early returns per GEMINI.md."'
Set-Content -Path ".gemini/commands/refactor/shrink.toml" -Value 'description = "Split class."`nprompt = "Extract logic to a new class and update Factory to stay under 100 lines."'
Set-Content -Path ".gemini/commands/refactor/sync.toml" -Value 'description = "Sync Contract."`nprompt = "Compare Service to Contract and fix missing methods."'
Set-Content -Path ".gemini/commands/test.toml" -Value 'description = "Quick tests."`nprompt = "Generate unit tests using the Factory for mocking."'

