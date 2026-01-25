# PerpetualCC Desktop App Distribution Guide

This document covers building and distributing the PerpetualCC desktop application for macOS.

## Table of Contents

1. [Overview](#overview)
2. [Technology Choice](#technology-choice)
3. [Building with PyInstaller](#building-with-pyinstaller)
4. [Building with Tauri](#building-with-tauri)
5. [Creating a DMG Installer](#creating-a-dmg-installer)
6. [Auto-Update with Sparkle](#auto-update-with-sparkle)
7. [Distribution Channels](#distribution-channels)
8. [Complete Build Pipeline](#complete-build-pipeline)

---

## Overview

### Distribution Goals

1. **Easy installation**: Drag-and-drop from DMG
2. **Auto-updates**: Users get new versions automatically
3. **CLI access**: Desktop app includes CLI (`pcc` command)
4. **Native experience**: Looks and feels like a macOS app
5. **Homebrew support**: `brew install --cask perpetualcc`

### Build Options

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **PyInstaller** | Pure Python, simple | Large bundle (~100MB) | MVP |
| **Nuitka** | Better performance | Complex build | Alternative |
| **Tauri** | Tiny (~10MB), modern | Rust required | Post-MVP |

---

## Technology Choice

### Recommended: PyInstaller for MVP

PyInstaller bundles Python + dependencies into a standalone .app:

- Fastest path to distribution
- No new languages required
- Well-documented
- Works with existing Python codebase

### Future: Tauri for 2.0

For a polished 2.0 release, consider Tauri:

- Much smaller bundle size
- Better native integration
- Modern web UI (React, Vue, Svelte)
- Requires Rust knowledge

---

## Building with PyInstaller

### Prerequisites

```bash
# Install PyInstaller
pip install pyinstaller

# Verify
pyinstaller --version
```

### Step 1: Create PyInstaller Spec File

Create `perpetualcc.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

# Project root
project_root = Path(__file__).parent

# Collect all package data
datas = [
    # Include any data files your app needs
    # (str(project_root / 'assets'), 'assets'),
]

# Hidden imports (modules not detected automatically)
hiddenimports = [
    'typer',
    'rich',
    'rich.console',
    'rich.panel',
    'rich.text',
    'pydantic',
    'anyio',
    'anyio._backends',
    'anyio._backends._asyncio',
    # Add claude-agent-sdk modules as needed
]

a = Analysis(
    ['perpetualcc/__main__.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'numpy',
        'pandas',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pcc',  # CLI executable name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # True for CLI, False for pure GUI
    disable_windowed_traceback=False,
    argv_emulation=True,  # Important for macOS
    target_arch=None,  # Will build for current architecture
    codesign_identity=None,  # We'll sign separately
    entitlements_file=None,
)

# Create the .app bundle
app = BUNDLE(
    exe,
    a.binaries,
    a.datas,
    name='PerpetualCC.app',
    icon='assets/icon.icns',  # Create this icon file
    bundle_identifier='com.perpetualcc.app',
    info_plist={
        'CFBundleName': 'PerpetualCC',
        'CFBundleDisplayName': 'PerpetualCC',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleIdentifier': 'com.perpetualcc.app',
        'CFBundleExecutable': 'pcc',
        'CFBundlePackageType': 'APPL',
        'LSMinimumSystemVersion': '13.0',
        'LSApplicationCategoryType': 'public.app-category.developer-tools',
        'NSHighResolutionCapable': True,
        'NSHumanReadableCopyright': 'Copyright © 2025 Chester Lee',
        # For menu bar / status bar app:
        # 'LSUIElement': True,
    },
)

# Also create a separate CLI-only build
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pcc-cli',  # CLI-only distribution
)
```

### Step 2: Create App Icon

```bash
# Create icon from PNG (requires Xcode)
# Start with a 1024x1024 PNG named icon.png

mkdir -p assets/icon.iconset

# Generate all sizes
sips -z 16 16 icon.png --out assets/icon.iconset/icon_16x16.png
sips -z 32 32 icon.png --out assets/icon.iconset/icon_16x16@2x.png
sips -z 32 32 icon.png --out assets/icon.iconset/icon_32x32.png
sips -z 64 64 icon.png --out assets/icon.iconset/icon_32x32@2x.png
sips -z 128 128 icon.png --out assets/icon.iconset/icon_128x128.png
sips -z 256 256 icon.png --out assets/icon.iconset/icon_128x128@2x.png
sips -z 256 256 icon.png --out assets/icon.iconset/icon_256x256.png
sips -z 512 512 icon.png --out assets/icon.iconset/icon_256x256@2x.png
sips -z 512 512 icon.png --out assets/icon.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out assets/icon.iconset/icon_512x512@2x.png

# Convert to .icns
iconutil -c icns assets/icon.iconset -o assets/icon.icns
```

### Step 3: Build the App

```bash
# Build using spec file
pyinstaller perpetualcc.spec --clean

# Output:
# dist/PerpetualCC.app  - macOS app bundle
# dist/pcc-cli/         - CLI-only distribution
```

### Step 4: Test the Build

```bash
# Test the app launches
open dist/PerpetualCC.app

# Test CLI from app bundle
dist/PerpetualCC.app/Contents/MacOS/pcc --version

# Test CLI-only build
dist/pcc-cli/pcc --version
```

### Build for Both Architectures

```bash
# Build for Intel (x86_64)
pyinstaller perpetualcc.spec --target-arch x86_64

# Build for Apple Silicon (arm64)
pyinstaller perpetualcc.spec --target-arch arm64

# Create universal binary (both architectures)
# Requires building both and using lipo to combine
mkdir -p dist/universal
lipo -create \
    dist/x86_64/PerpetualCC.app/Contents/MacOS/pcc \
    dist/arm64/PerpetualCC.app/Contents/MacOS/pcc \
    -output dist/universal/pcc

# Copy to universal app bundle
cp -R dist/arm64/PerpetualCC.app dist/universal/PerpetualCC.app
cp dist/universal/pcc dist/universal/PerpetualCC.app/Contents/MacOS/
```

---

## Building with Tauri

For post-MVP with modern web UI.

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Tauri CLI
cargo install tauri-cli

# Install Node.js (for frontend)
brew install node

# Verify
cargo tauri --version
```

### Project Structure

```
perpetualcc-desktop/
├── src-tauri/           # Rust backend
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       └── main.rs
├── src/                 # Frontend (React/Vue/Svelte)
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── index.html
```

### Tauri Configuration

`src-tauri/tauri.conf.json`:

```json
{
  "build": {
    "beforeBuildCommand": "npm run build",
    "beforeDevCommand": "npm run dev",
    "devPath": "http://localhost:5173",
    "distDir": "../dist"
  },
  "package": {
    "productName": "PerpetualCC",
    "version": "1.0.0"
  },
  "tauri": {
    "bundle": {
      "active": true,
      "category": "DeveloperTool",
      "copyright": "Copyright © 2025 Chester Lee",
      "identifier": "com.perpetualcc.app",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "macOS": {
        "minimumSystemVersion": "13.0",
        "entitlements": "./entitlements.plist",
        "exceptionDomain": null
      },
      "targets": ["dmg", "app"]
    },
    "security": {
      "csp": "default-src 'self'; connect-src 'self' https://api.anthropic.com"
    },
    "windows": [{
      "title": "PerpetualCC",
      "width": 1200,
      "height": 800,
      "minWidth": 800,
      "minHeight": 600,
      "resizable": true,
      "fullscreen": false
    }]
  }
}
```

### Build Tauri App

```bash
cd perpetualcc-desktop

# Development
cargo tauri dev

# Production build
cargo tauri build

# Output: src-tauri/target/release/bundle/
#   - dmg/PerpetualCC_1.0.0_aarch64.dmg
#   - macos/PerpetualCC.app
```

---

## Creating a DMG Installer

### Option 1: Using create-dmg (Recommended)

```bash
# Install create-dmg
brew install create-dmg

# Create DMG with nice styling
create-dmg \
    --volname "PerpetualCC" \
    --volicon "assets/icon.icns" \
    --background "assets/dmg-background.png" \
    --window-pos 200 120 \
    --window-size 660 400 \
    --icon-size 100 \
    --icon "PerpetualCC.app" 180 170 \
    --hide-extension "PerpetualCC.app" \
    --app-drop-link 480 170 \
    --no-internet-enable \
    "dist/PerpetualCC-0.1.0.dmg" \
    "dist/PerpetualCC.app"
```

### Option 2: Using hdiutil (Simple)

```bash
# Create simple DMG
hdiutil create \
    -volname "PerpetualCC" \
    -srcfolder "dist/PerpetualCC.app" \
    -ov \
    -format UDZO \
    "dist/PerpetualCC-0.1.0.dmg"
```

### DMG Background Image

Create a background image for the DMG window:

```
Dimensions: 660 x 400 pixels (or match window-size)
Format: PNG
Content:
  - App icon placeholder area (left side)
  - Arrow pointing right
  - Applications folder area (right side)
  - Optional: logo, instructions
```

---

## Auto-Update with Sparkle

Sparkle is the standard auto-update framework for macOS apps outside the App Store.

### Step 1: Add Sparkle Framework

```bash
# Download Sparkle
curl -L -o Sparkle.tar.xz \
    https://github.com/sparkle-project/Sparkle/releases/latest/download/Sparkle-2.x.tar.xz

# Extract
tar -xf Sparkle.tar.xz

# Copy framework to app
cp -R Sparkle.framework dist/PerpetualCC.app/Contents/Frameworks/
```

### Step 2: Configure Info.plist

Add to `Info.plist`:

```xml
<!-- Sparkle configuration -->
<key>SUFeedURL</key>
<string>https://perpetualcc.dev/appcast.xml</string>

<key>SUPublicEDKey</key>
<string>YOUR_ED25519_PUBLIC_KEY</string>

<key>SUEnableAutomaticChecks</key>
<true/>

<key>SUScheduledCheckInterval</key>
<integer>86400</integer>
```

### Step 3: Generate Signing Keys

```bash
# Generate Ed25519 key pair for Sparkle
./Sparkle.framework/bin/generate_keys

# Output:
# Private key saved to ~/Library/Sparkle/private/
# Public key: BASE64_PUBLIC_KEY
#
# Add public key to Info.plist as SUPublicEDKey
```

### Step 4: Create Appcast

Create `appcast.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>PerpetualCC Updates</title>
    <link>https://perpetualcc.dev/appcast.xml</link>
    <description>PerpetualCC release notes</description>
    <language>en</language>

    <item>
      <title>Version 1.0.0</title>
      <pubDate>Mon, 01 Jan 2025 12:00:00 +0000</pubDate>
      <sparkle:version>1.0.0</sparkle:version>
      <sparkle:shortVersionString>1.0.0</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>13.0</sparkle:minimumSystemVersion>
      <description>
        <![CDATA[
          <h2>What's New</h2>
          <ul>
            <li>Initial release</li>
            <li>Feature 1</li>
            <li>Feature 2</li>
          </ul>
        ]]>
      </description>
      <enclosure
        url="https://perpetualcc.dev/releases/PerpetualCC-1.0.0.dmg"
        sparkle:edSignature="SIGNATURE_HERE"
        length="12345678"
        type="application/octet-stream"/>
    </item>

  </channel>
</rss>
```

### Step 5: Sign Update Archive

```bash
# Sign the DMG for Sparkle
./Sparkle.framework/bin/sign_update \
    dist/PerpetualCC-1.0.0.dmg

# Output: sparkle:edSignature="BASE64_SIGNATURE"
# Add this to appcast.xml enclosure
```

### Step 6: Host Appcast

Upload to your website:

```bash
# Upload appcast.xml
scp appcast.xml server:perpetualcc.dev/appcast.xml

# Upload DMG
scp dist/PerpetualCC-1.0.0.dmg server:perpetualcc.dev/releases/
```

---

## Distribution Channels

### Channel 1: Website Download

```
https://perpetualcc.dev/download

Page content:
- Download button for latest DMG
- System requirements
- Installation instructions
- Release notes
- Link to GitHub releases
```

### Channel 2: GitHub Releases

```bash
# Upload to GitHub release
gh release upload v1.0.0 \
    dist/PerpetualCC-1.0.0.dmg \
    dist/pcc-cli.zip
```

### Channel 3: Homebrew Cask

See [HOMEBREW.md](./HOMEBREW.md) for cask setup.

```ruby
# Casks/perpetualcc.rb
cask "perpetualcc" do
  version "1.0.0"
  sha256 "..."

  url "https://github.com/USERNAME/perpetualcc/releases/download/v#{version}/PerpetualCC-#{version}.dmg"
  name "PerpetualCC"
  desc "Intelligent master agent for Claude Code orchestration"
  homepage "https://perpetualcc.dev"

  depends_on macos: ">= :ventura"

  app "PerpetualCC.app"
  binary "#{appdir}/PerpetualCC.app/Contents/MacOS/pcc"

  zap trash: [
    "~/Library/Application Support/PerpetualCC",
    "~/.perpetualcc",
  ]
end
```

---

## Complete Build Pipeline

### Makefile

Create `Makefile`:

```makefile
.PHONY: all clean build sign notarize dmg release

VERSION := $(shell grep -oP '__version__ = "\K[^"]+' perpetualcc/__init__.py)
APP_NAME := PerpetualCC
SIGNING_IDENTITY := "Developer ID Application: Your Name (TEAM_ID)"
KEYCHAIN_PROFILE := perpetualcc-notarize

all: release

clean:
	rm -rf build/ dist/ *.egg-info
	rm -f *.dmg *.zip

build: clean
	@echo "Building $(APP_NAME) v$(VERSION)..."
	pyinstaller perpetualcc.spec --clean
	@echo "Build complete: dist/$(APP_NAME).app"

sign: build
	@echo "Signing..."
	codesign --force --options runtime \
		--sign $(SIGNING_IDENTITY) \
		--timestamp --deep \
		--entitlements entitlements.plist \
		dist/$(APP_NAME).app
	codesign --verify --deep --strict dist/$(APP_NAME).app
	@echo "Signed successfully"

dmg: sign
	@echo "Creating DMG..."
	create-dmg \
		--volname "$(APP_NAME)" \
		--volicon "assets/icon.icns" \
		--window-pos 200 120 \
		--window-size 660 400 \
		--icon-size 100 \
		--icon "$(APP_NAME).app" 180 170 \
		--hide-extension "$(APP_NAME).app" \
		--app-drop-link 480 170 \
		"dist/$(APP_NAME)-$(VERSION).dmg" \
		"dist/$(APP_NAME).app"
	codesign --force --sign $(SIGNING_IDENTITY) \
		dist/$(APP_NAME)-$(VERSION).dmg
	@echo "DMG created: dist/$(APP_NAME)-$(VERSION).dmg"

notarize: dmg
	@echo "Notarizing..."
	xcrun notarytool submit dist/$(APP_NAME)-$(VERSION).dmg \
		--keychain-profile $(KEYCHAIN_PROFILE) \
		--wait
	xcrun stapler staple dist/$(APP_NAME)-$(VERSION).dmg
	@echo "Notarization complete"

verify: notarize
	@echo "Verifying..."
	spctl --assess --type open \
		--context context:primary-signature \
		--verbose dist/$(APP_NAME)-$(VERSION).dmg
	@echo "Verification complete"

release: verify
	@echo ""
	@echo "======================================"
	@echo "$(APP_NAME) v$(VERSION) ready for release"
	@echo "======================================"
	@echo ""
	@echo "Distribution file: dist/$(APP_NAME)-$(VERSION).dmg"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Upload to GitHub release"
	@echo "  2. Update Homebrew cask"
	@echo "  3. Update website download page"
	@echo "  4. Update Sparkle appcast.xml"
```

### Usage

```bash
# Full release build
make release

# Individual steps
make build      # Build only
make sign       # Build + sign
make dmg        # Build + sign + create DMG
make notarize   # Full pipeline with notarization
```

### GitHub Actions

`.github/workflows/build-desktop.yml`:

```yaml
name: Build Desktop App

on:
  push:
    tags:
      - 'v*'

jobs:
  build-macos:
    runs-on: macos-14  # Apple Silicon runner
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pyinstaller
          brew install create-dmg

      - name: Get version
        id: version
        run: |
          echo "version=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Create icon
        run: |
          # Assumes assets/icon.png exists
          mkdir -p assets/icon.iconset
          sips -z 1024 1024 assets/icon.png --out assets/icon.iconset/icon_512x512@2x.png
          sips -z 512 512 assets/icon.png --out assets/icon.iconset/icon_512x512.png
          sips -z 256 256 assets/icon.png --out assets/icon.iconset/icon_256x256.png
          iconutil -c icns assets/icon.iconset -o assets/icon.icns

      - name: Build app
        run: |
          pyinstaller perpetualcc.spec --clean

      - name: Import certificate
        env:
          CERTIFICATE_P12: ${{ secrets.APPLE_CERTIFICATE_P12 }}
          CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
        run: |
          echo "$CERTIFICATE_P12" | base64 --decode > cert.p12
          security create-keychain -p "" build.keychain
          security default-keychain -s build.keychain
          security unlock-keychain -p "" build.keychain
          security import cert.p12 -k build.keychain -P "$CERTIFICATE_PASSWORD" -T /usr/bin/codesign
          security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" build.keychain
          rm cert.p12

      - name: Sign and create DMG
        env:
          SIGNING_IDENTITY: ${{ secrets.SIGNING_IDENTITY }}
        run: |
          # Sign app
          codesign --force --options runtime \
            --sign "$SIGNING_IDENTITY" \
            --timestamp --deep \
            dist/PerpetualCC.app

          # Create DMG
          create-dmg \
            --volname "PerpetualCC" \
            --window-size 660 400 \
            --icon-size 100 \
            --icon "PerpetualCC.app" 180 170 \
            --app-drop-link 480 170 \
            "dist/PerpetualCC-${{ steps.version.outputs.version }}.dmg" \
            "dist/PerpetualCC.app"

          # Sign DMG
          codesign --force --sign "$SIGNING_IDENTITY" \
            "dist/PerpetualCC-${{ steps.version.outputs.version }}.dmg"

      - name: Notarize
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_APP_PASSWORD: ${{ secrets.APPLE_APP_PASSWORD }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
        run: |
          xcrun notarytool submit \
            "dist/PerpetualCC-${{ steps.version.outputs.version }}.dmg" \
            --apple-id "$APPLE_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --team-id "$APPLE_TEAM_ID" \
            --wait

          xcrun stapler staple \
            "dist/PerpetualCC-${{ steps.version.outputs.version }}.dmg"

      - name: Upload to release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh release upload v${{ steps.version.outputs.version }} \
            "dist/PerpetualCC-${{ steps.version.outputs.version }}.dmg"
```

---

## Quick Reference

### Build Commands

```bash
# PyInstaller build
pyinstaller perpetualcc.spec --clean

# Create DMG
create-dmg --volname "PerpetualCC" \
    --icon "PerpetualCC.app" 180 170 \
    --app-drop-link 480 170 \
    "PerpetualCC-1.0.0.dmg" \
    "dist/PerpetualCC.app"

# Sign
codesign --force --options runtime --sign "IDENTITY" --deep dist/PerpetualCC.app
codesign --force --sign "IDENTITY" dist/PerpetualCC-1.0.0.dmg

# Notarize
xcrun notarytool submit dist/PerpetualCC-1.0.0.dmg --keychain-profile "PROFILE" --wait

# Staple
xcrun stapler staple dist/PerpetualCC-1.0.0.dmg

# Verify
spctl --assess --type open --context context:primary-signature -v dist/PerpetualCC-1.0.0.dmg
```

### File Size Expectations

| Build Type | Expected Size |
|------------|---------------|
| PyInstaller .app | 80-150 MB |
| PyInstaller DMG | 40-80 MB (compressed) |
| Tauri .app | 5-15 MB |
| Tauri DMG | 3-10 MB (compressed) |
