# PerpetualCC Publishing Guide

Step-by-step instructions for publishing PerpetualCC releases. This guide is designed for a coding agent or developer to follow exactly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Publish Checklist](#pre-publish-checklist)
3. [Publishing to TestPyPI](#publishing-to-testpypi)
4. [Publishing to PyPI](#publishing-to-pypi)
5. [Creating GitHub Release](#creating-github-release)
6. [Post-Publish Verification](#post-publish-verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### One-Time Setup (Human Required)

These steps require human interaction and should be done once before first release.

#### 1. Create PyPI Account

```
ACTION: Human must complete

1. Go to https://pypi.org/account/register/
2. Create account with email verification
3. Enable 2FA (required for publishing)
4. Go to https://pypi.org/manage/account/token/
5. Create API token with scope "Entire account" (for first upload)
   - After first upload, create project-scoped token
6. Save token securely (starts with "pypi-")
```

#### 2. Create TestPyPI Account

```
ACTION: Human must complete

1. Go to https://test.pypi.org/account/register/
2. Create account (separate from PyPI)
3. Enable 2FA
4. Create API token at https://test.pypi.org/manage/account/token/
5. Save token securely (starts with "pypi-")
```

#### 3. Configure PyPI Credentials

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

Set permissions:
```bash
chmod 600 ~/.pypirc
```

#### 4. Install Publishing Tools

```bash
# Install build and publish tools
pip install --upgrade build twine pip-audit

# Verify installations
python -m build --version
twine --version
```

#### 5. Configure Git for Releases

```bash
# Ensure git is configured
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Install GitHub CLI (optional but recommended)
brew install gh
gh auth login
```

---

## Pre-Publish Checklist

Run these checks before every release.

### Step 1: Ensure Clean Working Directory

```bash
cd /path/to/perpetualcc

# Check for uncommitted changes
git status

# Expected output: "nothing to commit, working tree clean"
# If not clean, commit or stash changes first
```

### Step 2: Run All Tests

```bash
# Run full test suite
pytest

# Expected: All tests pass
# If failures, fix before proceeding
```

### Step 3: Run Linting

```bash
# Check for lint errors
ruff check perpetualcc/

# Fix auto-fixable issues
ruff check perpetualcc/ --fix

# Format code
ruff format perpetualcc/

# Verify no remaining issues
ruff check perpetualcc/
# Expected: No output (no errors)
```

### Step 4: Security Audit

```bash
# Audit dependencies for vulnerabilities
pip-audit

# Expected: No known vulnerabilities
# If vulnerabilities found, update dependencies or document exceptions
```

### Step 5: Update Version Numbers

**File 1: `perpetualcc/__init__.py`**

```python
"""PerpetualCC - An intelligent master agent that orchestrates Claude Code sessions 24/7."""

__version__ = "X.Y.Z"  # Update this
```

**File 2: `pyproject.toml`**

```toml
[project]
name = "perpetualcc"
version = "X.Y.Z"  # Update this (must match __init__.py)
```

**Verify versions match:**

```bash
# Extract versions and compare
INIT_VERSION=$(grep -oP '__version__ = "\K[^"]+' perpetualcc/__init__.py)
TOML_VERSION=$(grep -oP '^version = "\K[^"]+' pyproject.toml)

echo "Init version: $INIT_VERSION"
echo "TOML version: $TOML_VERSION"

# They must be identical
```

### Step 6: Update CHANGELOG.md

Create or update `CHANGELOG.md` in project root:

```markdown
# Changelog

All notable changes to PerpetualCC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [X.Y.Z] - YYYY-MM-DD

### Added
- Feature 1 description
- Feature 2 description

### Changed
- Change 1 description

### Fixed
- Bug fix 1 description

### Deprecated
- Deprecated feature (if any)

### Removed
- Removed feature (if any)

### Security
- Security fix (if any)
```

### Step 7: Commit Version Bump

```bash
# Stage changes
git add perpetualcc/__init__.py pyproject.toml CHANGELOG.md

# Commit
git commit -m "chore: bump version to X.Y.Z"

# Push to main/master
git push origin main
```

---

## Publishing to TestPyPI

Always test on TestPyPI before publishing to PyPI.

### Step 1: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info perpetualcc.egg-info/

# Verify clean
ls dist/ 2>/dev/null || echo "dist/ directory clean"
```

### Step 2: Build Package

```bash
# Build source distribution and wheel
python -m build

# Verify build artifacts
ls -la dist/

# Expected output:
# perpetualcc-X.Y.Z.tar.gz     (source distribution)
# perpetualcc-X.Y.Z-py3-none-any.whl  (wheel)
```

### Step 3: Validate Package

```bash
# Check package with twine
twine check dist/*

# Expected: "PASSED" for both files
```

### Step 4: Upload to TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Expected output:
# Uploading perpetualcc-X.Y.Z-py3-none-any.whl
# Uploading perpetualcc-X.Y.Z.tar.gz
# View at: https://test.pypi.org/project/perpetualcc/X.Y.Z/
```

### Step 5: Test Installation from TestPyPI

```bash
# Create fresh virtual environment for testing
python -m venv /tmp/test-pcc
source /tmp/test-pcc/bin/activate

# Install from TestPyPI
# Note: --extra-index-url needed because TestPyPI may not have all dependencies
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    perpetualcc

# Verify installation
pcc --version
# Expected: perpetualcc vX.Y.Z

pcc --help
# Expected: Help output displayed

# Cleanup
deactivate
rm -rf /tmp/test-pcc
```

---

## Publishing to PyPI

Only proceed after successful TestPyPI verification.

### Step 1: Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Expected output:
# Uploading perpetualcc-X.Y.Z-py3-none-any.whl
# Uploading perpetualcc-X.Y.Z.tar.gz
# View at: https://pypi.org/project/perpetualcc/X.Y.Z/
```

### Step 2: Verify on PyPI

```
ACTION: Human verification recommended

1. Visit https://pypi.org/project/perpetualcc/
2. Verify version X.Y.Z is displayed
3. Check project description renders correctly
4. Verify all metadata (author, license, links)
```

### Step 3: Test Installation from PyPI

```bash
# Create fresh virtual environment
python -m venv /tmp/verify-pcc
source /tmp/verify-pcc/bin/activate

# Install from PyPI
pip install perpetualcc

# Verify
pcc --version
pcc --help

# Test basic functionality
pcc version

# Cleanup
deactivate
rm -rf /tmp/verify-pcc
```

---

## Creating GitHub Release

### Step 1: Create Git Tag

```bash
# Create annotated tag
git tag -a vX.Y.Z -m "Release vX.Y.Z

Highlights:
- Feature 1
- Feature 2
- Bug fix 1

See CHANGELOG.md for full details."

# Push tag to remote
git push origin vX.Y.Z
```

### Step 2: Create GitHub Release

**Option A: Using GitHub CLI (Recommended)**

```bash
# Create release with auto-generated notes
gh release create vX.Y.Z \
    --title "PerpetualCC vX.Y.Z" \
    --notes-file CHANGELOG.md \
    dist/*

# Or with auto-generated notes from commits
gh release create vX.Y.Z \
    --title "PerpetualCC vX.Y.Z" \
    --generate-notes \
    dist/*
```

**Option B: Using GitHub Web UI**

```
ACTION: Human must complete

1. Go to https://github.com/USERNAME/perpetualcc/releases/new
2. Choose tag: vX.Y.Z
3. Release title: PerpetualCC vX.Y.Z
4. Description: Copy from CHANGELOG.md
5. Attach files:
   - dist/perpetualcc-X.Y.Z.tar.gz
   - dist/perpetualcc-X.Y.Z-py3-none-any.whl
6. Check "Set as the latest release"
7. Click "Publish release"
```

### Step 3: Verify GitHub Release

```bash
# List releases
gh release list

# View specific release
gh release view vX.Y.Z

# Verify assets are attached
gh release view vX.Y.Z --json assets
```

---

## Post-Publish Verification

### Verification Checklist

```bash
# 1. PyPI package accessible
pip index versions perpetualcc
# Expected: Shows X.Y.Z in list

# 2. Fresh install works
pip install perpetualcc==X.Y.Z
pcc --version
# Expected: vX.Y.Z

# 3. GitHub release accessible
gh release view vX.Y.Z
# Expected: Shows release details

# 4. Git tag exists
git ls-remote --tags origin | grep vX.Y.Z
# Expected: Shows tag reference
```

### Announce Release (Optional)

```
ACTION: Human may complete

1. Post to project's GitHub Discussions
2. Update project website (if exists)
3. Tweet/post on social media (if appropriate)
4. Notify relevant communities (Reddit, HN, Discord)
```

---

## Troubleshooting

### Error: "File already exists"

```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists.
```

**Cause**: Trying to upload a version that already exists on PyPI.

**Solution**: PyPI does not allow re-uploading the same version. You must:
1. Increment the version number (even for patches: X.Y.Z → X.Y.Z+1)
2. Rebuild and re-upload

### Error: "Invalid distribution file"

```
InvalidDistribution: Cannot find file dist/perpetualcc-X.Y.Z.tar.gz
```

**Cause**: Build artifacts missing or path incorrect.

**Solution**:
```bash
# Clean and rebuild
rm -rf dist/ build/
python -m build
ls dist/  # Verify files exist
```

### Error: "Authentication failed"

```
HTTPError: 403 Forbidden from https://upload.pypi.org/legacy/
Invalid or non-existent authentication information.
```

**Cause**: Invalid or expired API token.

**Solution**:
1. Generate new token at https://pypi.org/manage/account/token/
2. Update `~/.pypirc` with new token
3. Ensure username is `__token__` (literal string)

### Error: "Package name already taken"

```
HTTPError: 400 Bad Request
The name 'perpetualcc' is too similar to an existing project.
```

**Cause**: Package name conflicts with existing PyPI package.

**Solution**:
1. Check https://pypi.org/project/PACKAGE_NAME/
2. If genuinely taken, choose different name
3. If you own it, ensure you're logged into correct account

### Error: "Metadata validation failed"

```
error: The description failed to render for 'text/x-rst'.
```

**Cause**: README has invalid RST formatting (if using RST).

**Solution**:
```bash
# Check README rendering
python -m readme_renderer README.md

# Or switch to Markdown in pyproject.toml
# readme = "README.md"  # Already using markdown
```

### Error: "Missing required metadata"

```
error: metadata is missing required fields: author
```

**Cause**: pyproject.toml missing required fields.

**Solution**: Ensure all required fields are present:
```toml
[project]
name = "perpetualcc"
version = "X.Y.Z"
description = "Description here"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "Your Name" }]
```

---

## Quick Reference Commands

```bash
# Full release sequence (copy-paste ready)

# 1. Pre-checks
git status
pytest
ruff check perpetualcc/ --fix
ruff format perpetualcc/

# 2. Version bump (edit files manually)
# perpetualcc/__init__.py: __version__ = "X.Y.Z"
# pyproject.toml: version = "X.Y.Z"

# 3. Commit and tag
git add -A
git commit -m "chore: release vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main --tags

# 4. Build
rm -rf dist/ build/
python -m build

# 5. Test upload
twine upload --repository testpypi dist/*

# 6. Production upload
twine upload dist/*

# 7. GitHub release
gh release create vX.Y.Z --generate-notes dist/*

# 8. Verify
pip install --upgrade perpetualcc
pcc --version
```
