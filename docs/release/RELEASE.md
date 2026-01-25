# PerpetualCC Release Guide

This document provides a comprehensive overview of PerpetualCC's release strategy, distribution channels, and versioning policy.

## Table of Contents

1. [Release Philosophy](#release-philosophy)
2. [Version Numbering](#version-numbering)
3. [Distribution Channels](#distribution-channels)
4. [Release Phases](#release-phases)
5. [Dependencies](#dependencies)
6. [Pre-Release Checklist](#pre-release-checklist)
7. [Related Documents](#related-documents)

---

## Release Philosophy

PerpetualCC follows these principles for releases:

1. **Progressive Enhancement**: Start with PyPI, expand to Homebrew, then desktop app
2. **Developer-First**: Prioritize CLI experience for the MVP
3. **Zero-Friction Install**: Each channel should require minimal steps
4. **Backwards Compatibility**: Breaking changes only in major versions
5. **Security-First**: All releases signed and verified

---

## Version Numbering

We follow [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
  0.1.0       - Initial development release
  0.2.0-alpha - Alpha pre-release
  0.2.0-beta  - Beta pre-release
  0.2.0-rc.1  - Release candidate
  0.2.0       - Stable release
  1.0.0       - First stable major release
```

### Version Meaning

| Version Part | When to Increment |
|--------------|-------------------|
| MAJOR | Breaking API changes, incompatible config changes |
| MINOR | New features, backwards-compatible |
| PATCH | Bug fixes, security patches |

### Pre-release Tags

| Tag | Meaning | Stability |
|-----|---------|-----------|
| `alpha` | Feature incomplete, unstable | Low |
| `beta` | Feature complete, testing needed | Medium |
| `rc.N` | Release candidate N, production-ready candidate | High |

### Version Locations

Version must be updated in these files:

1. `perpetualcc/__init__.py` - `__version__ = "X.Y.Z"`
2. `pyproject.toml` - `version = "X.Y.Z"`

---

## Distribution Channels

### Channel Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PerpetualCC Distribution Channels                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │    PyPI      │   │   Homebrew   │   │   Website    │                │
│  │              │   │              │   │              │                │
│  │  pip install │   │ brew install │   │  DMG + .app  │                │
│  │  perpetualcc │   │ perpetualcc  │   │  download    │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│         │                  │                  │                          │
│         │                  ├─────────┬────────┤                          │
│         │                  │         │        │                          │
│         ▼                  ▼         ▼        ▼                          │
│  ┌──────────────┐   ┌──────────┐ ┌──────┐ ┌──────┐                     │
│  │   TestPyPI   │   │ Formula  │ │ Cask │ │GitHub│                     │
│  │   (staging)  │   │  (CLI)   │ │(GUI) │ │Releas│                     │
│  └──────────────┘   └──────────┘ └──────┘ └──────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Channel Details

| Channel | Type | Target Audience | Priority |
|---------|------|-----------------|----------|
| PyPI | Python Package | Python developers, CI/CD | P0 (MVP) |
| TestPyPI | Staging | Testing releases | P0 (MVP) |
| Homebrew Formula | CLI Package | macOS CLI users | P1 |
| GitHub Releases | Binary + Source | All users | P1 |
| Homebrew Cask | Desktop App | macOS GUI users | P2 |
| Website DMG | Desktop App | General users | P2 |

---

## Release Phases

### Phase 1: PyPI Only (MVP)

**Trigger**: Phases 1-3 of implementation complete

**Channels**:
- TestPyPI (for testing)
- PyPI (production)

**User Installation**:
```bash
pip install perpetualcc
# or
pipx install perpetualcc
```

**Requirements**:
- [ ] All unit tests passing
- [ ] README.md complete with usage examples
- [ ] LICENSE file present
- [ ] pyproject.toml metadata complete

### Phase 2: Add Homebrew (Beta)

**Trigger**: Phases 4-6 of implementation complete

**Channels**:
- PyPI
- Homebrew Formula (custom tap)
- GitHub Releases

**User Installation**:
```bash
# Option 1: PyPI
pip install perpetualcc

# Option 2: Homebrew
brew tap chesterlee/perpetualcc
brew install perpetualcc
```

**Requirements**:
- [ ] Phase 1 requirements
- [ ] homebrew-perpetualcc repository created
- [ ] Formula tested on macOS 13+ (Ventura, Sonoma, Sequoia)
- [ ] GitHub Release with changelog

### Phase 3: Desktop App (1.0)

**Trigger**: Phases 7-9 complete + Desktop UI implemented

**Channels**:
- All Phase 2 channels
- Homebrew Cask
- Website download (DMG)
- Auto-update (Sparkle)

**User Installation**:
```bash
# Option 1-2: Same as Phase 2

# Option 3: Homebrew Cask
brew install --cask chesterlee/perpetualcc/perpetualcc

# Option 4: Direct download
# Download from https://perpetualcc.dev/download
```

**Requirements**:
- [ ] Phase 2 requirements
- [ ] Apple Developer Account ($99/year)
- [ ] Developer ID certificate
- [ ] App signed and notarized
- [ ] DMG installer created
- [ ] Website with download page
- [ ] Sparkle appcast.xml hosted

---

## Dependencies

### Runtime Dependencies

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| claude-agent-sdk | >=0.1.0 | Claude Code SDK | Yes |
| typer | >=0.12.0 | CLI framework | Yes |
| rich | >=13.0.0 | Terminal UI | Yes |
| anyio | >=4.0.0 | Async utilities | Yes |
| pydantic | >=2.0.0 | Data validation | Yes |

### Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| gemini | google-genai>=1.0.0 | Gemini API brain |
| ollama | ollama>=0.4.0 | Local LLM brain |
| knowledge | chromadb, sentence-transformers, networkx, tree-sitter | RAG + code graph |
| faiss | faiss-cpu>=1.8.0 | Alternative vector store |
| dev | pytest, pytest-asyncio, ruff | Development |

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11 | 3.12 |
| macOS | 13.0 (Ventura) | 14.0+ (Sonoma) |
| RAM | 4GB | 8GB |
| Disk | 500MB | 2GB (with knowledge) |

### External Requirements

| Service | Required For | How to Get |
|---------|--------------|------------|
| Anthropic API Key | Claude Code sessions | https://console.anthropic.com |
| Gemini API Key | Gemini brain (optional) | https://aistudio.google.com |
| Ollama | Local LLM brain (optional) | https://ollama.ai |

---

## Pre-Release Checklist

### Code Quality

- [ ] All tests passing: `pytest`
- [ ] No lint errors: `ruff check perpetualcc/`
- [ ] Code formatted: `ruff format perpetualcc/`
- [ ] Type hints complete (optional but recommended)
- [ ] No `console.log` or debug statements
- [ ] No hardcoded secrets or API keys

### Documentation

- [ ] README.md updated with:
  - [ ] Installation instructions for all channels
  - [ ] Quick start guide
  - [ ] Feature overview
  - [ ] Requirements
- [ ] CHANGELOG.md updated with:
  - [ ] Version number and date
  - [ ] Added features
  - [ ] Changed behaviors
  - [ ] Fixed bugs
  - [ ] Breaking changes (if any)
- [ ] CLAUDE.md updated (if architecture changed)

### Version Bump

- [ ] `perpetualcc/__init__.py` version updated
- [ ] `pyproject.toml` version updated
- [ ] Git tag created: `git tag -a vX.Y.Z -m "Release X.Y.Z"`

### Security

- [ ] Dependencies audited: `pip-audit`
- [ ] No known vulnerabilities
- [ ] Secrets removed from history (if accidentally committed)

### Testing

- [ ] Unit tests: `pytest tests/unit/`
- [ ] Integration tests: `pytest tests/integration/`
- [ ] Manual testing on clean environment
- [ ] Tested on target Python versions (3.11, 3.12)
- [ ] Tested on target macOS versions

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [PUBLISHING.md](./PUBLISHING.md) | Step-by-step publishing instructions |
| [HOMEBREW.md](./HOMEBREW.md) | Homebrew formula and cask setup |
| [SIGNING.md](./SIGNING.md) | macOS code signing and notarization |
| [DESKTOP_APP.md](./DESKTOP_APP.md) | Desktop app build and distribution |

---

## Quick Reference

### Release Commands

```bash
# 1. Run tests
pytest

# 2. Lint and format
ruff check perpetualcc/ --fix
ruff format perpetualcc/

# 3. Build package
python -m build

# 4. Test on TestPyPI
twine upload --repository testpypi dist/*

# 5. Install from TestPyPI and verify
pip install --index-url https://test.pypi.org/simple/ perpetualcc

# 6. Upload to PyPI
twine upload dist/*

# 7. Create GitHub release
gh release create vX.Y.Z --generate-notes

# 8. Update Homebrew formula
# (See HOMEBREW.md)
```

### Emergency Rollback

If a release has critical issues:

1. **PyPI**: Yank the release (does not delete, prevents new installs)
   ```bash
   # Cannot fully delete, but can yank
   # Go to PyPI project page → Manage → Release → Options → Yank
   ```

2. **Homebrew**: Revert formula to previous version
   ```bash
   cd homebrew-perpetualcc
   git revert HEAD
   git push
   ```

3. **GitHub**: Delete release and tag
   ```bash
   gh release delete vX.Y.Z --yes
   git push --delete origin vX.Y.Z
   git tag -d vX.Y.Z
   ```

4. **Notify users**: Post to GitHub Discussions/Issues about the issue
