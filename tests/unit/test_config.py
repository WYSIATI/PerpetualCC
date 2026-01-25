"""Unit tests for configuration module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from perpetualcc.config import (
    BrainConfig,
    BrainType,
    DataConfig,
    GeminiConfig,
    KnowledgeConfig,
    NotificationsConfig,
    OllamaConfig,
    OutputConfig,
    OutputVerbosity,
    PermissionsConfig,
    PerpetualCCConfig,
    SessionsConfig,
    generate_default_config,
    load_config,
    save_config,
)


class TestBrainType:
    """Tests for BrainType enum."""

    def test_brain_type_values(self):
        """Test brain type enum values."""
        assert BrainType.RULE_BASED.value == "rule_based"
        assert BrainType.GEMINI.value == "gemini"
        assert BrainType.OLLAMA.value == "ollama"

    def test_brain_type_from_string(self):
        """Test creating brain type from string."""
        assert BrainType("rule_based") == BrainType.RULE_BASED
        assert BrainType("gemini") == BrainType.GEMINI
        assert BrainType("ollama") == BrainType.OLLAMA


class TestGeminiConfig:
    """Tests for GeminiConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GeminiConfig()
        assert config.api_key is None
        assert config.model == "gemini-2.0-flash"

    def test_get_api_key_direct(self):
        """Test getting API key when set directly."""
        config = GeminiConfig(api_key="test-key")
        assert config.get_api_key() == "test-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """Test getting API key from environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        config = GeminiConfig()
        assert config.get_api_key() == "env-key"

    def test_get_api_key_env_prefix(self, monkeypatch):
        """Test getting API key with env: prefix."""
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-key")
        config = GeminiConfig(api_key="env:MY_CUSTOM_KEY")
        assert config.get_api_key() == "custom-key"

    def test_get_api_key_missing_env(self, monkeypatch):
        """Test getting API key when env var is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = GeminiConfig(api_key="env:NONEXISTENT_VAR")
        assert config.get_api_key() is None


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OllamaConfig()
        assert config.host == "http://localhost:11434"
        assert config.model == "llama3.2"
        assert config.timeout == 120


class TestBrainConfig:
    """Tests for BrainConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BrainConfig()
        assert config.type == BrainType.RULE_BASED
        assert config.confidence_threshold == 0.7
        assert isinstance(config.gemini, GeminiConfig)
        assert isinstance(config.ollama, OllamaConfig)


class TestPerpetualCCConfig:
    """Tests for PerpetualCCConfig."""

    def test_default_factory(self):
        """Test creating default configuration."""
        config = PerpetualCCConfig.default()
        assert isinstance(config.brain, BrainConfig)
        assert isinstance(config.sessions, SessionsConfig)
        assert isinstance(config.permissions, PermissionsConfig)
        assert isinstance(config.notifications, NotificationsConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.knowledge, KnowledgeConfig)
        assert isinstance(config.data, DataConfig)

    def test_default_brain_type(self):
        """Test default brain type is rule-based."""
        config = PerpetualCCConfig.default()
        assert config.brain.type == BrainType.RULE_BASED

    def test_default_confidence_threshold(self):
        """Test default confidence threshold."""
        config = PerpetualCCConfig.default()
        assert config.brain.confidence_threshold == 0.7


class TestConfigIO:
    """Tests for config loading and saving."""

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns defaults."""
        config = load_config(Path("/nonexistent/path/config.toml"))
        assert config.brain.type == BrainType.RULE_BASED

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            # Create a custom config
            config = PerpetualCCConfig(
                brain=BrainConfig(
                    type=BrainType.GEMINI,
                    confidence_threshold=0.8,
                    gemini=GeminiConfig(model="gemini-pro"),
                ),
            )

            # Save it
            save_config(config, config_path)
            assert config_path.exists()

            # Load it back
            loaded = load_config(config_path)
            assert loaded.brain.type == BrainType.GEMINI
            assert loaded.brain.confidence_threshold == 0.8
            assert loaded.brain.gemini.model == "gemini-pro"

    def test_save_creates_directory(self):
        """Test that save creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.toml"

            config = PerpetualCCConfig.default()
            save_config(config, config_path)

            assert config_path.exists()
            assert config_path.parent.is_dir()

    def test_load_partial_config(self):
        """Test loading config with only some values set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            # Write minimal config
            config_path.write_text("""
[brain]
type = "ollama"
""")

            loaded = load_config(config_path)
            assert loaded.brain.type == BrainType.OLLAMA
            # Other values should be defaults
            assert loaded.brain.confidence_threshold == 0.7

    def test_load_invalid_toml(self):
        """Test loading invalid TOML returns defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"

            # Write invalid TOML
            config_path.write_text("this is not valid toml [[[")

            # Should return defaults, not raise
            loaded = load_config(config_path)
            assert loaded.brain.type == BrainType.RULE_BASED


class TestGenerateDefaultConfig:
    """Tests for default config generation."""

    def test_generate_default_config(self):
        """Test generating default config string."""
        config_str = generate_default_config()

        # Should be valid TOML
        assert "[brain]" in config_str
        assert "[brain.gemini]" in config_str
        assert "[brain.ollama]" in config_str
        assert "[sessions]" in config_str
        assert "[permissions]" in config_str
        assert "[notifications]" in config_str
        assert "[output]" in config_str

    def test_generated_config_is_parseable(self):
        """Test that generated config can be parsed."""
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        config_str = generate_default_config()
        # Should not raise
        parsed = tomllib.loads(config_str)
        assert "brain" in parsed


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_values(self):
        """Test default output configuration."""
        config = OutputConfig()
        assert config.verbosity == OutputVerbosity.NORMAL
        assert config.show_timestamps is True
        assert config.color_enabled is True


class TestSessionsConfig:
    """Tests for SessionsConfig."""

    def test_default_values(self):
        """Test default sessions configuration."""
        config = SessionsConfig()
        assert config.max_concurrent == 5
        assert config.auto_resume is True
        assert config.persist_history is True


class TestPermissionsConfig:
    """Tests for PermissionsConfig."""

    def test_default_values(self):
        """Test default permissions configuration."""
        config = PermissionsConfig()
        assert config.auto_approve_low_risk is True
        assert "src/" in config.safe_directories
        assert "tests/" in config.safe_directories

    def test_default_blocked_commands(self):
        """Test default blocked commands."""
        config = PermissionsConfig()
        assert "rm -rf /" in config.blocked_commands
        assert "sudo" in config.blocked_commands
