# PerpetualCC Homebrew Distribution Guide

This document covers setting up and maintaining Homebrew distribution for PerpetualCC, including both Formula (CLI) and Cask (desktop app).

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Creating the Tap Repository](#creating-the-tap-repository)
4. [Creating a Homebrew Formula](#creating-a-homebrew-formula)
5. [Testing the Formula](#testing-the-formula)
6. [Publishing Formula Updates](#publishing-formula-updates)
7. [Creating a Homebrew Cask](#creating-a-homebrew-cask)
8. [Automation](#automation)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Homebrew Terminology

| Term | Description |
|------|-------------|
| **Tap** | Third-party repository of Homebrew packages |
| **Formula** | Ruby script for building/installing CLI tools |
| **Cask** | Ruby script for installing macOS GUI applications |
| **Bottle** | Pre-built binary package (faster install) |

### Distribution Structure

```
github.com/USERNAME/homebrew-perpetualcc/
├── Formula/
│   └── perpetualcc.rb          # CLI formula
├── Casks/
│   └── perpetualcc.rb          # Desktop app cask (Phase 2)
├── README.md
└── LICENSE
```

### User Installation Commands

```bash
# Add the tap
brew tap USERNAME/perpetualcc

# Install CLI (Formula)
brew install perpetualcc

# Install Desktop App (Cask) - Phase 2
brew install --cask perpetualcc
```

---

## Prerequisites

### One-Time Setup (Human Required)

#### 1. GitHub Account with Repository Access

```
ACTION: Human must complete

1. Ensure GitHub account is active
2. Have push access to create new repository
3. Have gh CLI authenticated: gh auth status
```

#### 2. Install Homebrew Development Tools

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Verify Homebrew
brew --version

# Install helpful tools
brew install gh          # GitHub CLI
brew install jq          # JSON processor
pip install homebrew-pypi-poet  # Generate Python resource stanzas
```

#### 3. Understand Homebrew Python Packaging

Homebrew Python formulas:
- Create a virtualenv in the Cellar
- Install all PyPI dependencies from source
- Symlink the executable to `/usr/local/bin/` (Intel) or `/opt/homebrew/bin/` (Apple Silicon)

---

## Creating the Tap Repository

### Step 1: Create GitHub Repository

**Option A: Using GitHub CLI**

```bash
# Create repository
gh repo create homebrew-perpetualcc \
    --public \
    --description "Homebrew tap for PerpetualCC" \
    --clone

cd homebrew-perpetualcc
```

**Option B: Using GitHub Web UI**

```
ACTION: Human must complete

1. Go to https://github.com/new
2. Repository name: homebrew-perpetualcc
   (MUST start with "homebrew-" for tap shorthand)
3. Description: Homebrew tap for PerpetualCC
4. Public repository
5. Initialize with README
6. Add MIT License
7. Create repository
8. Clone locally: git clone git@github.com:USERNAME/homebrew-perpetualcc.git
```

### Step 2: Set Up Repository Structure

```bash
cd homebrew-perpetualcc

# Create directories
mkdir -p Formula Casks

# Create README
cat > README.md << 'EOF'
# Homebrew Tap for PerpetualCC

This tap contains Homebrew formulas and casks for [PerpetualCC](https://github.com/USERNAME/perpetualcc).

## Installation

### CLI Tool (Formula)

```bash
brew tap USERNAME/perpetualcc
brew install perpetualcc
```

### Desktop App (Cask)

```bash
brew tap USERNAME/perpetualcc
brew install --cask perpetualcc
```

## Updating

```bash
brew update
brew upgrade perpetualcc
```

## Uninstalling

```bash
brew uninstall perpetualcc
brew untap USERNAME/perpetualcc
```

## Available Packages

| Package | Type | Description |
|---------|------|-------------|
| perpetualcc | Formula | CLI for orchestrating Claude Code sessions |
| perpetualcc | Cask | Desktop application (coming soon) |

## Issues

Report issues at: https://github.com/USERNAME/perpetualcc/issues
EOF

# Commit
git add .
git commit -m "Initial tap setup"
git push origin main
```

---

## Creating a Homebrew Formula

### Step 1: Generate Resource Stanzas

First, ensure PerpetualCC is published to PyPI, then generate dependency stanzas:

```bash
# Install poet
pip install homebrew-pypi-poet

# Generate resources for perpetualcc and all dependencies
poet perpetualcc > /tmp/resources.rb

# View generated resources
cat /tmp/resources.rb
```

The output will look like:

```ruby
resource "typer" do
  url "https://files.pythonhosted.org/packages/.../typer-0.12.0.tar.gz"
  sha256 "abc123..."
end

resource "rich" do
  url "https://files.pythonhosted.org/packages/.../rich-13.0.0.tar.gz"
  sha256 "def456..."
end
# ... more resources
```

### Step 2: Get Package Checksum

```bash
# Get SHA256 of the PyPI tarball
VERSION="0.1.0"
curl -sL "https://files.pythonhosted.org/packages/source/p/perpetualcc/perpetualcc-${VERSION}.tar.gz" | shasum -a 256
```

### Step 3: Create the Formula

Create `Formula/perpetualcc.rb`:

```ruby
class Perpetualcc < Formula
  include Language::Python::Virtualenv

  desc "Intelligent master agent that orchestrates Claude Code sessions 24/7"
  homepage "https://github.com/USERNAME/perpetualcc"
  url "https://files.pythonhosted.org/packages/source/p/perpetualcc/perpetualcc-0.1.0.tar.gz"
  sha256 "PASTE_SHA256_HERE"
  license "MIT"

  depends_on "python@3.12"

  # Paste all resource stanzas from poet here
  resource "typer" do
    url "https://files.pythonhosted.org/packages/.../typer-0.12.0.tar.gz"
    sha256 "..."
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/.../rich-13.0.0.tar.gz"
    sha256 "..."
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/.../click-8.1.7.tar.gz"
    sha256 "..."
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/.../anyio-4.0.0.tar.gz"
    sha256 "..."
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/.../pydantic-2.0.0.tar.gz"
    sha256 "..."
  end

  # Add ALL transitive dependencies here
  # Use: poet perpetualcc --resources

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/pcc --version")
    assert_match "PerpetualCC", shell_output("#{bin}/pcc --help")
  end
end
```

### Step 4: Complete Formula Template

Here's a complete formula template with common patterns:

```ruby
class Perpetualcc < Formula
  include Language::Python::Virtualenv

  desc "Intelligent master agent that orchestrates Claude Code sessions 24/7"
  homepage "https://github.com/USERNAME/perpetualcc"
  url "https://files.pythonhosted.org/packages/source/p/perpetualcc/perpetualcc-0.1.0.tar.gz"
  sha256 "CHECKSUM_HERE"
  license "MIT"
  head "https://github.com/USERNAME/perpetualcc.git", branch: "main"

  # Bottles (pre-built binaries) - added after first bottle build
  # bottle do
  #   sha256 cellar: :any_skip_relocation, arm64_sonoma: "..."
  #   sha256 cellar: :any_skip_relocation, ventura: "..."
  # end

  depends_on "python@3.12"

  # ============================================
  # RESOURCES: Paste output from `poet perpetualcc --resources`
  # ============================================

  # Direct dependencies
  resource "claude-agent-sdk" do
    url "https://files.pythonhosted.org/packages/..."
    sha256 "..."
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/..."
    sha256 "..."
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/..."
    sha256 "..."
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/..."
    sha256 "..."
  end

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/..."
    sha256 "..."
  end

  # Transitive dependencies (from poet output)
  # ... add all dependencies here ...

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      PerpetualCC requires an Anthropic API key.
      Set it in your environment:
        export ANTHROPIC_API_KEY="your-key-here"

      For optional features:
        - Gemini brain: export GEMINI_API_KEY="your-key"
        - Ollama brain: install Ollama from https://ollama.ai

      Quick start:
        pcc start /path/to/project --task "Your task here"

      Documentation:
        https://github.com/USERNAME/perpetualcc#readme
    EOS
  end

  test do
    # Test version output
    assert_match version.to_s, shell_output("#{bin}/pcc --version")

    # Test help output
    assert_match "orchestrates Claude Code sessions", shell_output("#{bin}/pcc --help")

    # Test that CLI loads without error
    system "#{bin}/pcc", "version"
  end
end
```

---

## Testing the Formula

### Step 1: Local Testing

```bash
cd homebrew-perpetualcc

# Audit the formula for issues
brew audit --strict --online Formula/perpetualcc.rb

# Test the formula
brew test Formula/perpetualcc.rb

# Install from local formula
brew install --build-from-source Formula/perpetualcc.rb

# Verify installation
pcc --version
pcc --help

# Uninstall
brew uninstall perpetualcc
```

### Step 2: Test Installation Flow

```bash
# Simulate user experience

# 1. Tap the repository (from local path for testing)
brew tap-new test/perpetualcc
cp Formula/perpetualcc.rb "$(brew --repository test/perpetualcc)/Formula/"

# 2. Install
brew install test/perpetualcc/perpetualcc

# 3. Verify
which pcc
pcc --version

# 4. Cleanup
brew uninstall perpetualcc
brew untap test/perpetualcc
```

### Step 3: Comprehensive Test

```bash
# Run Homebrew's complete test suite
brew test-bot --only-tap-syntax

# Audit for style issues
brew audit --new-formula Formula/perpetualcc.rb

# Style check
brew style Formula/perpetualcc.rb
```

---

## Publishing Formula Updates

### Step 1: Commit and Push Formula

```bash
cd homebrew-perpetualcc

# Add formula
git add Formula/perpetualcc.rb

# Commit
git commit -m "perpetualcc 0.1.0 (new formula)

Intelligent master agent that orchestrates Claude Code sessions 24/7"

# Push
git push origin main
```

### Step 2: Test Public Installation

```bash
# Tap from GitHub
brew tap USERNAME/perpetualcc

# Install
brew install perpetualcc

# Verify
pcc --version
```

### Updating for New Releases

When releasing a new version:

```bash
# 1. Get new checksum
NEW_VERSION="0.2.0"
curl -sL "https://files.pythonhosted.org/packages/source/p/perpetualcc/perpetualcc-${NEW_VERSION}.tar.gz" | shasum -a 256

# 2. Update formula
# - Change url version
# - Update sha256
# - Regenerate resources if dependencies changed: poet perpetualcc --resources

# 3. Test locally
brew reinstall --build-from-source Formula/perpetualcc.rb
pcc --version

# 4. Commit and push
git add Formula/perpetualcc.rb
git commit -m "perpetualcc ${NEW_VERSION}

- Feature 1
- Feature 2"
git push origin main
```

---

## Creating a Homebrew Cask

For the desktop application (Phase 2).

### Cask Requirements

- Signed and notarized .app or .dmg
- Hosted on publicly accessible URL
- Consistent download URL pattern

### Step 1: Create the Cask

Create `Casks/perpetualcc.rb`:

```ruby
cask "perpetualcc" do
  version "1.0.0"
  sha256 "CHECKSUM_OF_DMG"

  url "https://github.com/USERNAME/perpetualcc/releases/download/v#{version}/PerpetualCC-#{version}.dmg"
  name "PerpetualCC"
  desc "Intelligent master agent for Claude Code orchestration"
  homepage "https://github.com/USERNAME/perpetualcc"

  # Minimum macOS version
  depends_on macos: ">= :ventura"

  # Install the app
  app "PerpetualCC.app"

  # Also install CLI from the app bundle
  binary "#{appdir}/PerpetualCC.app/Contents/MacOS/pcc"

  # Uninstall instructions
  uninstall quit: "com.perpetualcc.app"

  # Cleanup on uninstall
  zap trash: [
    "~/Library/Application Support/PerpetualCC",
    "~/Library/Preferences/com.perpetualcc.app.plist",
    "~/Library/Caches/com.perpetualcc.app",
    "~/.perpetualcc",
  ]

  caveats <<~EOS
    PerpetualCC requires an Anthropic API key.

    You can set it in:
    1. The app's Settings menu
    2. Environment: export ANTHROPIC_API_KEY="your-key"

    The CLI is also available as `pcc` in your terminal.
  EOS
end
```

### Step 2: Test the Cask

```bash
# Audit
brew audit --cask Casks/perpetualcc.rb

# Install
brew install --cask Casks/perpetualcc.rb

# Verify
open -a PerpetualCC
pcc --version

# Uninstall
brew uninstall --cask perpetualcc
```

---

## Automation

### GitHub Action for Formula Updates

Create `.github/workflows/update-formula.yml` in the main perpetualcc repo:

```yaml
name: Update Homebrew Formula

on:
  release:
    types: [published]

jobs:
  update-formula:
    runs-on: macos-latest
    steps:
      - name: Get release info
        id: release
        run: |
          echo "version=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Calculate SHA256
        id: sha
        run: |
          VERSION="${{ steps.release.outputs.version }}"
          SHA=$(curl -sL "https://files.pythonhosted.org/packages/source/p/perpetualcc/perpetualcc-${VERSION}.tar.gz" | shasum -a 256 | cut -d' ' -f1)
          echo "sha256=${SHA}" >> $GITHUB_OUTPUT

      - name: Update formula
        env:
          GH_TOKEN: ${{ secrets.HOMEBREW_TAP_TOKEN }}
        run: |
          # Clone tap repo
          gh repo clone USERNAME/homebrew-perpetualcc /tmp/tap
          cd /tmp/tap

          # Update formula
          VERSION="${{ steps.release.outputs.version }}"
          SHA="${{ steps.sha.outputs.sha256 }}"

          sed -i '' "s/url \".*perpetualcc-.*.tar.gz\"/url \"https:\/\/files.pythonhosted.org\/packages\/source\/p\/perpetualcc\/perpetualcc-${VERSION}.tar.gz\"/" Formula/perpetualcc.rb
          sed -i '' "s/sha256 \".*\"/sha256 \"${SHA}\"/" Formula/perpetualcc.rb

          # Commit and push
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add Formula/perpetualcc.rb
          git commit -m "perpetualcc ${VERSION}"
          git push
```

### Required Secrets

```
ACTION: Human must complete

1. Go to main perpetualcc repo → Settings → Secrets → Actions
2. Add secret: HOMEBREW_TAP_TOKEN
   - Personal access token with repo scope
   - Used to push to homebrew-perpetualcc repo
```

---

## Troubleshooting

### Error: "No available formula"

```
Error: No available formula with the name "perpetualcc"
```

**Cause**: Tap not added or formula name mismatch.

**Solution**:
```bash
brew tap USERNAME/perpetualcc
brew search perpetualcc
```

### Error: "Resource not found"

```
Error: perpetualcc: resource "dependency-name" not found
```

**Cause**: Missing dependency in resource stanzas.

**Solution**:
```bash
# Regenerate all resources
poet perpetualcc --resources > /tmp/resources.rb
# Add missing resources to formula
```

### Error: "SHA256 mismatch"

```
Error: SHA256 mismatch
Expected: abc123...
Actual: def456...
```

**Cause**: PyPI package was re-uploaded or URL changed.

**Solution**:
```bash
# Get correct SHA256
curl -sL "URL_FROM_FORMULA" | shasum -a 256
# Update formula with correct sha256
```

### Error: "Python version conflict"

```
Error: python@3.12 is not installed
```

**Cause**: Python dependency not installed.

**Solution**:
```bash
brew install python@3.12
brew reinstall perpetualcc
```

### Error: "virtualenv_install_with_resources failed"

**Cause**: Python package doesn't have proper setup.py or pyproject.toml.

**Solution**:
Ensure pyproject.toml has proper build system:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Testing Tips

```bash
# Verbose install for debugging
brew install --verbose --debug perpetualcc

# Check what would be installed
brew info perpetualcc

# See formula source
brew cat perpetualcc

# Force reinstall
brew reinstall perpetualcc

# Check logs
cat $(brew --prefix)/var/log/perpetualcc*
```

---

## Quick Reference

### Formula Update Checklist

```bash
# 1. Publish new version to PyPI first (see PUBLISHING.md)

# 2. Get new SHA256
VERSION="X.Y.Z"
curl -sL "https://files.pythonhosted.org/packages/source/p/perpetualcc/perpetualcc-${VERSION}.tar.gz" | shasum -a 256

# 3. Update Formula/perpetualcc.rb
# - Update url with new version
# - Update sha256

# 4. If dependencies changed, regenerate resources
poet perpetualcc --resources

# 5. Test
brew audit --strict Formula/perpetualcc.rb
brew install --build-from-source Formula/perpetualcc.rb
pcc --version

# 6. Commit and push
git add Formula/perpetualcc.rb
git commit -m "perpetualcc ${VERSION}"
git push origin main
```

### User Commands Reference

```bash
# Add tap
brew tap USERNAME/perpetualcc

# Install CLI
brew install perpetualcc

# Install desktop app
brew install --cask perpetualcc

# Upgrade
brew upgrade perpetualcc

# Uninstall
brew uninstall perpetualcc

# Remove tap
brew untap USERNAME/perpetualcc

# Get info
brew info perpetualcc

# Check for issues
brew doctor
```
