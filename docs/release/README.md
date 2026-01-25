# PerpetualCC Release Documentation

This directory contains comprehensive documentation for releasing and distributing PerpetualCC.

## Documentation Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [RELEASE.md](./RELEASE.md) | Release strategy overview | Start here for release planning |
| [PUBLISHING.md](./PUBLISHING.md) | Step-by-step PyPI publishing | Every CLI release |
| [HOMEBREW.md](./HOMEBREW.md) | Homebrew formula & cask setup | Setting up/updating Homebrew |
| [SIGNING.md](./SIGNING.md) | macOS code signing & notarization | Desktop app releases |
| [DESKTOP_APP.md](./DESKTOP_APP.md) | Desktop app build & distribution | Building GUI application |

## Quick Start by Release Type

### CLI-Only Release (Phase 1)

```bash
# 1. Read: RELEASE.md (version numbering, checklist)
# 2. Follow: PUBLISHING.md (step-by-step)
```

### CLI + Homebrew Release (Phase 2)

```bash
# 1. Complete CLI release first
# 2. Read: HOMEBREW.md (formula setup)
# 3. Follow: HOMEBREW.md → "Publishing Formula Updates"
```

### Desktop App Release (Phase 3)

```bash
# 1. Read: DESKTOP_APP.md (build options)
# 2. Read: SIGNING.md (certificate setup - human required)
# 3. Follow: DESKTOP_APP.md → "Complete Build Pipeline"
# 4. Follow: HOMEBREW.md → "Creating a Homebrew Cask"
```

## Human Actions Required

Some steps cannot be automated and require human interaction:

### One-Time Setup

| Action | Document | Section |
|--------|----------|---------|
| Create PyPI account | PUBLISHING.md | Prerequisites |
| Create TestPyPI account | PUBLISHING.md | Prerequisites |
| Configure ~/.pypirc | PUBLISHING.md | Prerequisites |
| Create GitHub homebrew-perpetualcc repo | HOMEBREW.md | Creating the Tap Repository |
| Enroll in Apple Developer Program | SIGNING.md | Apple Developer Account |
| Create App-Specific Password | SIGNING.md | Create App-Specific Password |
| Generate Developer ID certificate | SIGNING.md | Certificate Setup |
| Store notarization credentials | SIGNING.md | Store Credentials Securely |

### Per-Release Actions

| Action | Document | Section |
|--------|----------|---------|
| Update version numbers | PUBLISHING.md | Update Version Numbers |
| Update CHANGELOG.md | PUBLISHING.md | Update CHANGELOG.md |
| Verify PyPI upload | PUBLISHING.md | Verify on PyPI |
| Create GitHub release (web UI option) | PUBLISHING.md | Using GitHub Web UI |

## Release Checklist Summary

### Before Every Release

- [ ] All tests pass: `pytest`
- [ ] No lint errors: `ruff check perpetualcc/`
- [ ] Code formatted: `ruff format perpetualcc/`
- [ ] Version bumped in `__init__.py` and `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Changes committed and pushed

### CLI Release

- [ ] Build: `python -m build`
- [ ] Test upload: `twine upload --repository testpypi dist/*`
- [ ] Verify: `pip install --index-url https://test.pypi.org/simple/ perpetualcc`
- [ ] Production: `twine upload dist/*`
- [ ] GitHub release: `gh release create vX.Y.Z dist/*`

### Homebrew Release

- [ ] Get SHA256: `curl -sL "PyPI_URL" | shasum -a 256`
- [ ] Update Formula/perpetualcc.rb
- [ ] Test: `brew install --build-from-source Formula/perpetualcc.rb`
- [ ] Push to tap repository

### Desktop Release

- [ ] Build: `pyinstaller perpetualcc.spec`
- [ ] Sign: `codesign --force --options runtime --sign "IDENTITY" dist/PerpetualCC.app`
- [ ] Create DMG: `create-dmg ... dist/PerpetualCC.dmg dist/PerpetualCC.app`
- [ ] Sign DMG: `codesign --force --sign "IDENTITY" dist/PerpetualCC.dmg`
- [ ] Notarize: `xcrun notarytool submit dist/PerpetualCC.dmg --wait`
- [ ] Staple: `xcrun stapler staple dist/PerpetualCC.dmg`
- [ ] Verify: `spctl --assess --type open ... dist/PerpetualCC.dmg`
- [ ] Upload to GitHub release
- [ ] Update Homebrew cask
- [ ] Update Sparkle appcast.xml

## File Structure

```
docs/release/
├── README.md           # This file (index)
├── RELEASE.md          # Release strategy & versioning
├── PUBLISHING.md       # PyPI publishing guide
├── HOMEBREW.md         # Homebrew formula & cask
├── SIGNING.md          # macOS signing & notarization
└── DESKTOP_APP.md      # Desktop app distribution
```

## Support

- **Issues**: https://github.com/chesterlee/perpetualcc/issues
- **Discussions**: https://github.com/chesterlee/perpetualcc/discussions
