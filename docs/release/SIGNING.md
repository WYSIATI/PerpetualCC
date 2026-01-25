# macOS Code Signing and Notarization Guide

This document covers Apple code signing and notarization for PerpetualCC, required for distributing macOS applications outside the App Store.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Certificate Setup](#certificate-setup)
4. [Code Signing](#code-signing)
5. [Notarization](#notarization)
6. [Stapling](#stapling)
7. [Verification](#verification)
8. [Automation](#automation)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Sign and Notarize?

Starting with macOS 10.15 (Catalina), Apple requires all software distributed outside the App Store to be:

1. **Signed** with a Developer ID certificate
2. **Notarized** by Apple (scanned for malware)
3. **Stapled** with the notarization ticket

Without these, users see: **"App is damaged and can't be opened"**

### Process Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    macOS Signing & Notarization Flow                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Build App     Sign App       Upload to      Wait for       Staple      │
│  ─────────     ────────       Apple          Apple          ──────      │
│                                                                          │
│  [.app]  ──►  codesign   ──►  notarytool ──► Notarization ──► stapler  │
│                --sign         submit         Service         staple     │
│                                                                          │
│                                    │              │                      │
│                                    ▼              ▼                      │
│                              [Upload .zip]   [Scan for                   │
│                               or .dmg]        malware]                   │
│                                                   │                      │
│                                                   ▼                      │
│                                            [Approved or                  │
│                                             Rejected]                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Time Estimates

| Step | Duration |
|------|----------|
| Code signing | Seconds |
| Notarization upload | 1-5 minutes |
| Apple processing | 5-15 minutes (varies) |
| Stapling | Seconds |

---

## Prerequisites

### One-Time Setup (Human Required)

#### 1. Apple Developer Account

```
ACTION: Human must complete

1. Go to https://developer.apple.com/programs/
2. Enroll in Apple Developer Program ($99/year USD)
3. Complete enrollment (may take 24-48 hours)
4. Verify account at https://developer.apple.com/account/
```

#### 2. Create App-Specific Password

```
ACTION: Human must complete

1. Go to https://appleid.apple.com/
2. Sign in with Apple ID (same as Developer Account)
3. Navigate to: Sign-In and Security → App-Specific Passwords
4. Click "Generate an app-specific password"
5. Label: "PerpetualCC Notarization"
6. Save the generated password securely (format: xxxx-xxxx-xxxx-xxxx)
```

#### 3. Install Xcode Command Line Tools

```bash
# Install Xcode CLI tools (required for codesign, notarytool)
xcode-select --install

# Verify installation
xcrun --version
codesign --version
xcrun notarytool --version
```

---

## Certificate Setup

### Step 1: Create Developer ID Certificate

**Option A: Using Xcode (Recommended)**

```
ACTION: Human must complete

1. Open Xcode
2. Go to: Xcode → Settings → Accounts
3. Select your Apple ID → Manage Certificates
4. Click "+" → Developer ID Application
5. Certificate is created and installed in Keychain
```

**Option B: Using Developer Portal**

```
ACTION: Human must complete

1. Go to https://developer.apple.com/account/resources/certificates/list
2. Click "+" to create new certificate
3. Select "Developer ID Application"
4. Follow instructions to create CSR (Certificate Signing Request)
5. Upload CSR, download certificate
6. Double-click .cer file to install in Keychain
```

### Step 2: Verify Certificate Installation

```bash
# List Developer ID certificates in Keychain
security find-identity -v -p codesigning | grep "Developer ID Application"

# Expected output:
# 1) ABCD1234... "Developer ID Application: Your Name (TEAM_ID)"
#    1 valid identities found

# Note your Team ID (10-character alphanumeric, e.g., ABC1234DEF)
```

### Step 3: Store Credentials Securely

For automation, store credentials in Keychain:

```bash
# Store notarization credentials in Keychain
xcrun notarytool store-credentials "perpetualcc-notarize" \
    --apple-id "your@email.com" \
    --team-id "YOUR_TEAM_ID" \
    --password "xxxx-xxxx-xxxx-xxxx"

# Verify stored credentials
xcrun notarytool store-credentials --list
```

---

## Code Signing

### Signing a .app Bundle

```bash
# Variables
APP_PATH="path/to/PerpetualCC.app"
SIGNING_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"

# Sign the app with hardened runtime (required for notarization)
codesign --force \
    --options runtime \
    --sign "$SIGNING_IDENTITY" \
    --timestamp \
    --entitlements entitlements.plist \
    "$APP_PATH"

# Sign all nested executables and frameworks
codesign --force \
    --options runtime \
    --sign "$SIGNING_IDENTITY" \
    --timestamp \
    --deep \
    "$APP_PATH"
```

### Entitlements File

Create `entitlements.plist` for your app:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- Required for hardened runtime -->
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <false/>

    <!-- Allow network access -->
    <key>com.apple.security.network.client</key>
    <true/>

    <!-- Allow reading user-selected files -->
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>

    <!-- For Python-based apps: allow loading plugins -->
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>

    <!-- For apps using JIT (not usually needed) -->
    <!-- <key>com.apple.security.cs.allow-jit</key>
    <true/> -->
</dict>
</plist>
```

### Signing a DMG

```bash
# Variables
DMG_PATH="path/to/PerpetualCC-1.0.0.dmg"
SIGNING_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"

# Sign the DMG
codesign --force \
    --sign "$SIGNING_IDENTITY" \
    --timestamp \
    "$DMG_PATH"
```

### Verify Signing

```bash
# Verify app signature
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
# Expected: "valid on disk" and "satisfies its Designated Requirement"

# Check signature details
codesign -dv --verbose=4 "$APP_PATH"

# Verify DMG signature
codesign --verify --verbose "$DMG_PATH"
```

---

## Notarization

### Step 1: Prepare for Upload

For .app files, create a ZIP archive:

```bash
# Create ZIP for notarization (preserves code signatures)
ditto -c -k --keepParent "$APP_PATH" "PerpetualCC.zip"
```

For DMG files, use the DMG directly.

### Step 2: Submit for Notarization

**Option A: Using Stored Credentials (Recommended)**

```bash
# Submit using stored credentials
xcrun notarytool submit "PerpetualCC.zip" \
    --keychain-profile "perpetualcc-notarize" \
    --wait

# Or for DMG
xcrun notarytool submit "$DMG_PATH" \
    --keychain-profile "perpetualcc-notarize" \
    --wait
```

**Option B: Using Direct Credentials**

```bash
# Submit with credentials inline
xcrun notarytool submit "PerpetualCC.zip" \
    --apple-id "your@email.com" \
    --password "xxxx-xxxx-xxxx-xxxx" \
    --team-id "YOUR_TEAM_ID" \
    --wait
```

### Step 3: Monitor Progress

The `--wait` flag blocks until notarization completes. For async submission:

```bash
# Submit without waiting
SUBMISSION_ID=$(xcrun notarytool submit "PerpetualCC.zip" \
    --keychain-profile "perpetualcc-notarize" \
    --output-format json | jq -r '.id')

echo "Submission ID: $SUBMISSION_ID"

# Check status
xcrun notarytool info "$SUBMISSION_ID" \
    --keychain-profile "perpetualcc-notarize"

# Wait for completion
xcrun notarytool wait "$SUBMISSION_ID" \
    --keychain-profile "perpetualcc-notarize"
```

### Step 4: Review Notarization Log

Always check the log, even on success:

```bash
# Get notarization log
xcrun notarytool log "$SUBMISSION_ID" \
    --keychain-profile "perpetualcc-notarize" \
    notarization_log.json

# View log
cat notarization_log.json | jq .

# Check for issues
cat notarization_log.json | jq '.issues'
```

---

## Stapling

After successful notarization, staple the ticket to the app/DMG:

### Staple .app

```bash
# Staple to app bundle
xcrun stapler staple "$APP_PATH"

# Expected output: "The staple and validate action worked!"
```

### Staple DMG

```bash
# Staple to DMG
xcrun stapler staple "$DMG_PATH"

# Expected output: "The staple and validate action worked!"
```

### Verify Stapling

```bash
# Verify staple
xcrun stapler validate "$APP_PATH"
xcrun stapler validate "$DMG_PATH"

# Check Gatekeeper acceptance
spctl --assess --type execute --verbose "$APP_PATH"
# Expected: "accepted source=Notarized Developer ID"

spctl --assess --type open --context context:primary-signature --verbose "$DMG_PATH"
# Expected: "accepted source=Notarized Developer ID"
```

---

## Verification

### Complete Verification Checklist

```bash
#!/bin/bash
# verification_check.sh

APP_PATH="PerpetualCC.app"
DMG_PATH="PerpetualCC-1.0.0.dmg"

echo "=== Code Signature Verification ==="

echo -n "App signature: "
if codesign --verify --deep --strict "$APP_PATH" 2>/dev/null; then
    echo "✅ Valid"
else
    echo "❌ Invalid"
fi

echo -n "DMG signature: "
if codesign --verify "$DMG_PATH" 2>/dev/null; then
    echo "✅ Valid"
else
    echo "❌ Invalid"
fi

echo ""
echo "=== Notarization Verification ==="

echo -n "App stapled: "
if xcrun stapler validate "$APP_PATH" 2>/dev/null; then
    echo "✅ Stapled"
else
    echo "❌ Not stapled"
fi

echo -n "DMG stapled: "
if xcrun stapler validate "$DMG_PATH" 2>/dev/null; then
    echo "✅ Stapled"
else
    echo "❌ Not stapled"
fi

echo ""
echo "=== Gatekeeper Verification ==="

echo -n "App Gatekeeper: "
if spctl --assess --type execute "$APP_PATH" 2>/dev/null; then
    echo "✅ Accepted"
else
    echo "❌ Rejected"
fi

echo -n "DMG Gatekeeper: "
if spctl --assess --type open --context context:primary-signature "$DMG_PATH" 2>/dev/null; then
    echo "✅ Accepted"
else
    echo "❌ Rejected"
fi

echo ""
echo "=== Certificate Details ==="
codesign -dv --verbose=2 "$APP_PATH" 2>&1 | grep -E "Authority|TeamIdentifier|Timestamp"
```

---

## Automation

### Complete Signing Script

Create `sign_and_notarize.sh`:

```bash
#!/bin/bash
set -e

# Configuration
APP_NAME="PerpetualCC"
APP_PATH="dist/${APP_NAME}.app"
DMG_PATH="dist/${APP_NAME}-${VERSION}.dmg"
SIGNING_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
KEYCHAIN_PROFILE="perpetualcc-notarize"
ENTITLEMENTS="entitlements.plist"

# Get version from app or argument
VERSION="${1:-$(defaults read "${APP_PATH}/Contents/Info.plist" CFBundleShortVersionString)}"

echo "========================================"
echo "Signing and Notarizing ${APP_NAME} v${VERSION}"
echo "========================================"

# Step 1: Sign the app
echo ""
echo "Step 1: Signing app..."
codesign --force \
    --options runtime \
    --sign "$SIGNING_IDENTITY" \
    --timestamp \
    --entitlements "$ENTITLEMENTS" \
    --deep \
    "$APP_PATH"

codesign --verify --deep --strict "$APP_PATH"
echo "✅ App signed"

# Step 2: Create DMG
echo ""
echo "Step 2: Creating DMG..."
if command -v create-dmg &> /dev/null; then
    create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "${APP_NAME}.app" 150 190 \
        --app-drop-link 450 185 \
        "$DMG_PATH" \
        "$APP_PATH"
else
    hdiutil create -volname "$APP_NAME" \
        -srcfolder "$APP_PATH" \
        -ov -format UDZO \
        "$DMG_PATH"
fi
echo "✅ DMG created"

# Step 3: Sign the DMG
echo ""
echo "Step 3: Signing DMG..."
codesign --force \
    --sign "$SIGNING_IDENTITY" \
    --timestamp \
    "$DMG_PATH"
echo "✅ DMG signed"

# Step 4: Notarize
echo ""
echo "Step 4: Notarizing (this may take several minutes)..."
xcrun notarytool submit "$DMG_PATH" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --wait

echo "✅ Notarization complete"

# Step 5: Staple
echo ""
echo "Step 5: Stapling..."
xcrun stapler staple "$DMG_PATH"
echo "✅ Stapled"

# Step 6: Verify
echo ""
echo "Step 6: Verifying..."
spctl --assess --type open --context context:primary-signature --verbose "$DMG_PATH"

echo ""
echo "========================================"
echo "✅ ${APP_NAME} v${VERSION} ready for distribution"
echo "   Output: ${DMG_PATH}"
echo "========================================"
```

### GitHub Actions Workflow

```yaml
name: Build, Sign, and Notarize macOS App

on:
  release:
    types: [published]

jobs:
  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pyinstaller

      - name: Build app
        run: |
          pyinstaller --name PerpetualCC \
            --windowed \
            --icon assets/icon.icns \
            --add-data "assets:assets" \
            perpetualcc/__main__.py

      - name: Import certificates
        env:
          CERTIFICATE_P12: ${{ secrets.APPLE_CERTIFICATE_P12 }}
          CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
          KEYCHAIN_PASSWORD: ${{ secrets.KEYCHAIN_PASSWORD }}
        run: |
          # Create temporary keychain
          security create-keychain -p "$KEYCHAIN_PASSWORD" build.keychain
          security default-keychain -s build.keychain
          security unlock-keychain -p "$KEYCHAIN_PASSWORD" build.keychain

          # Import certificate
          echo "$CERTIFICATE_P12" | base64 --decode > certificate.p12
          security import certificate.p12 -k build.keychain \
            -P "$CERTIFICATE_PASSWORD" \
            -T /usr/bin/codesign
          rm certificate.p12

          # Allow codesign to access keychain
          security set-key-partition-list -S apple-tool:,apple:,codesign: \
            -s -k "$KEYCHAIN_PASSWORD" build.keychain

      - name: Sign and notarize
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_APP_PASSWORD: ${{ secrets.APPLE_APP_PASSWORD }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
          SIGNING_IDENTITY: ${{ secrets.SIGNING_IDENTITY }}
        run: |
          # Sign
          codesign --force --options runtime \
            --sign "$SIGNING_IDENTITY" \
            --timestamp --deep \
            "dist/PerpetualCC.app"

          # Create DMG
          hdiutil create -volname "PerpetualCC" \
            -srcfolder "dist/PerpetualCC.app" \
            -ov -format UDZO \
            "dist/PerpetualCC-${{ github.ref_name }}.dmg"

          # Sign DMG
          codesign --force --sign "$SIGNING_IDENTITY" \
            "dist/PerpetualCC-${{ github.ref_name }}.dmg"

          # Notarize
          xcrun notarytool submit "dist/PerpetualCC-${{ github.ref_name }}.dmg" \
            --apple-id "$APPLE_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --team-id "$APPLE_TEAM_ID" \
            --wait

          # Staple
          xcrun stapler staple "dist/PerpetualCC-${{ github.ref_name }}.dmg"

      - name: Upload to release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh release upload ${{ github.ref_name }} \
            "dist/PerpetualCC-${{ github.ref_name }}.dmg"
```

### Required GitHub Secrets

```
ACTION: Human must complete

Add these secrets to GitHub repository (Settings → Secrets → Actions):

1. APPLE_CERTIFICATE_P12
   - Export Developer ID certificate from Keychain as .p12
   - Base64 encode: base64 -i certificate.p12 | pbcopy
   - Paste as secret value

2. APPLE_CERTIFICATE_PASSWORD
   - Password used when exporting .p12

3. KEYCHAIN_PASSWORD
   - Any secure password for temporary keychain

4. APPLE_ID
   - Your Apple ID email

5. APPLE_APP_PASSWORD
   - App-specific password for notarization

6. APPLE_TEAM_ID
   - 10-character Team ID

7. SIGNING_IDENTITY
   - Full signing identity string:
     "Developer ID Application: Your Name (TEAM_ID)"
```

---

## Troubleshooting

### Error: "The signature is invalid"

```
PerpetualCC.app: invalid signature
```

**Causes:**
1. App modified after signing
2. Nested code not signed
3. Resources modified

**Solution:**
```bash
# Re-sign with --deep flag
codesign --force --deep --options runtime \
    --sign "$SIGNING_IDENTITY" \
    --timestamp \
    "$APP_PATH"
```

### Error: "Notarization failed"

```
Error: The software was not notarized.
```

**Solution:**
```bash
# Get detailed log
xcrun notarytool log "$SUBMISSION_ID" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    log.json

# Common issues in log:
# - "The signature does not include a secure timestamp"
#   → Add --timestamp to codesign
# - "The executable does not have the hardened runtime enabled"
#   → Add --options runtime to codesign
# - "The signature of the binary is invalid"
#   → Re-sign the binary
```

### Error: "App is damaged"

Users see: **"PerpetualCC.app is damaged and can't be opened"**

**Cause:** Not properly signed/notarized OR user downloaded with quarantine.

**For users (temporary workaround):**
```bash
# Remove quarantine attribute
xattr -cr /Applications/PerpetualCC.app
```

**For developers:** Ensure complete signing and notarization process.

### Error: "No identity found"

```
errSecInternalComponent
no identity found
```

**Cause:** Certificate not in Keychain or not trusted.

**Solution:**
```bash
# List available identities
security find-identity -v -p codesigning

# If certificate missing, reinstall from Apple Developer portal
# If certificate untrusted, open Keychain Access → trust certificate
```

### Error: "Unable to get notarization credentials"

```
Error: Unable to get notarization credentials.
```

**Cause:** Stored credentials invalid or expired.

**Solution:**
```bash
# Re-store credentials
xcrun notarytool store-credentials "perpetualcc-notarize" \
    --apple-id "your@email.com" \
    --team-id "YOUR_TEAM_ID" \
    --password "NEW_APP_SPECIFIC_PASSWORD"
```

### Checking Notarization Status

```bash
# Get submission history
xcrun notarytool history --keychain-profile "perpetualcc-notarize"

# Get info on specific submission
xcrun notarytool info "$SUBMISSION_ID" \
    --keychain-profile "perpetualcc-notarize"

# Possible statuses:
# - "In Progress" - Still processing
# - "Accepted" - Success
# - "Invalid" - Failed, check log
# - "Rejected" - Contains malware/policy violation
```

---

## Quick Reference

### Commands Cheatsheet

```bash
# Sign app
codesign --force --options runtime --sign "IDENTITY" --timestamp --deep APP.app

# Sign DMG
codesign --force --sign "IDENTITY" --timestamp APP.dmg

# Verify signature
codesign --verify --deep --strict --verbose=2 APP.app

# Store credentials
xcrun notarytool store-credentials "PROFILE" --apple-id EMAIL --team-id TEAM --password PASS

# Submit for notarization
xcrun notarytool submit APP.dmg --keychain-profile "PROFILE" --wait

# Get notarization log
xcrun notarytool log SUBMISSION_ID --keychain-profile "PROFILE" log.json

# Staple
xcrun stapler staple APP.dmg

# Verify notarization
spctl --assess --type open --context context:primary-signature -v APP.dmg

# List certificates
security find-identity -v -p codesigning
```
