"""Configuration management for PerpetualCC.

Handles loading, saving, and validating configuration from TOML files.
Configuration is stored at ~/.perpetualcc/config.toml by default.

Example configuration:
    [brain]
    type = "rule_based"  # "rule_based" | "gemini" | "ollama"
    confidence_threshold = 0.7

    [brain.gemini]
    api_key = "env:GEMINI_API_KEY"  # or direct key
    model = "gemini-2.0-flash"

    [brain.ollama]
    host = "http://localhost:11434"
    model = "deepseek-coder:33b"

    [sessions]
    max_concurrent = 5
    auto_resume = true

    [permissions]
    auto_approve_low_risk = true
    safe_directories = ["src/", "tests/", "lib/", "app/"]

    [notifications]
    enabled = true
    sound = true

    [output]
    verbosity = "normal"  # "quiet" | "normal" | "verbose"
    show_timestamps = true
    color_enabled = true
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Python 3.11+ has tomllib in stdlib (read-only)
# For Python 3.10, use tomli package as fallback
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        # If tomli is not installed, we'll handle it gracefully
        tomllib = None  # type: ignore


class BrainType(Enum):
    """Available brain types."""

    RULE_BASED = "rule_based"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class OutputVerbosity(Enum):
    """Output verbosity levels."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


@dataclass
class GeminiConfig:
    """Configuration for Gemini brain."""

    api_key: str | None = None  # Can be "env:VAR_NAME" or direct key
    model: str = "gemini-2.0-flash"

    def get_api_key(self) -> str | None:
        """Get the actual API key, resolving env: prefix."""
        if not self.api_key:
            # Try default environment variable
            return os.environ.get("GEMINI_API_KEY")
        if self.api_key.startswith("env:"):
            var_name = self.api_key[4:]
            return os.environ.get(var_name)
        return self.api_key


@dataclass
class OllamaConfig:
    """Configuration for Ollama brain."""

    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: int = 120  # seconds


@dataclass
class BrainConfig:
    """Configuration for the brain layer."""

    type: BrainType = BrainType.RULE_BASED
    confidence_threshold: float = 0.7
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class SessionsConfig:
    """Configuration for session management."""

    max_concurrent: int = 5
    auto_resume: bool = True
    persist_history: bool = True
    checkpoint_retention_days: int = 30


@dataclass
class PermissionsConfig:
    """Configuration for permission handling."""

    auto_approve_low_risk: bool = True
    safe_directories: list[str] = field(
        default_factory=lambda: ["src/", "tests/", "lib/", "app/", "packages/"]
    )
    blocked_commands: list[str] = field(
        default_factory=lambda: ["rm -rf /", "sudo", "curl | sh"]
    )


@dataclass
class NotificationsConfig:
    """Configuration for notifications."""

    enabled: bool = True
    sound: bool = True
    on_question: bool = True
    on_complete: bool = True
    on_rate_limit: bool = True
    on_error: bool = True


@dataclass
class OutputConfig:
    """Configuration for CLI output."""

    verbosity: OutputVerbosity = OutputVerbosity.NORMAL
    show_timestamps: bool = True
    color_enabled: bool = True
    show_file_changes: bool = True
    show_thinking: bool = True


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge engine."""

    embedding_provider: str = "local"  # "local" | "gemini" | "hybrid"
    local_model: str = "nomic-embed-text"
    auto_index: bool = True


@dataclass
class DataConfig:
    """Configuration for data storage."""

    directory: str = "~/.perpetualcc/data"
    checkpoint_retention_days: int = 30
    episode_retention_days: int = 90


@dataclass
class PerpetualCCConfig:
    """Complete PerpetualCC configuration."""

    brain: BrainConfig = field(default_factory=BrainConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def default(cls) -> PerpetualCCConfig:
        """Create default configuration."""
        return cls()


# Default configuration file paths
DEFAULT_CONFIG_DIR = Path.home() / ".perpetualcc"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_DATA_DIR = DEFAULT_CONFIG_DIR / "data"


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists.

    Returns:
        Path to the configuration directory
    """
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CONFIG_DIR


def ensure_data_dir() -> Path:
    """Ensure the data directory exists.

    Returns:
        Path to the data directory
    """
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


def _parse_brain_config(data: dict[str, Any]) -> BrainConfig:
    """Parse brain configuration from dict."""
    brain_type_str = data.get("type", "rule_based")
    try:
        brain_type = BrainType(brain_type_str)
    except ValueError:
        brain_type = BrainType.RULE_BASED

    gemini_data = data.get("gemini", {})
    gemini_config = GeminiConfig(
        api_key=gemini_data.get("api_key"),
        model=gemini_data.get("model", "gemini-2.0-flash"),
    )

    ollama_data = data.get("ollama", {})
    ollama_config = OllamaConfig(
        host=ollama_data.get("host", "http://localhost:11434"),
        model=ollama_data.get("model", "llama3.2"),
        timeout=ollama_data.get("timeout", 120),
    )

    return BrainConfig(
        type=brain_type,
        confidence_threshold=data.get("confidence_threshold", 0.7),
        gemini=gemini_config,
        ollama=ollama_config,
    )


def _parse_sessions_config(data: dict[str, Any]) -> SessionsConfig:
    """Parse sessions configuration from dict."""
    return SessionsConfig(
        max_concurrent=data.get("max_concurrent", 5),
        auto_resume=data.get("auto_resume", True),
        persist_history=data.get("persist_history", True),
        checkpoint_retention_days=data.get("checkpoint_retention_days", 30),
    )


def _parse_permissions_config(data: dict[str, Any]) -> PermissionsConfig:
    """Parse permissions configuration from dict."""
    return PermissionsConfig(
        auto_approve_low_risk=data.get("auto_approve_low_risk", True),
        safe_directories=data.get(
            "safe_directories", ["src/", "tests/", "lib/", "app/", "packages/"]
        ),
        blocked_commands=data.get(
            "blocked_commands", ["rm -rf /", "sudo", "curl | sh"]
        ),
    )


def _parse_notifications_config(data: dict[str, Any]) -> NotificationsConfig:
    """Parse notifications configuration from dict."""
    return NotificationsConfig(
        enabled=data.get("enabled", True),
        sound=data.get("sound", True),
        on_question=data.get("on_question", True),
        on_complete=data.get("on_complete", True),
        on_rate_limit=data.get("on_rate_limit", True),
        on_error=data.get("on_error", True),
    )


def _parse_output_config(data: dict[str, Any]) -> OutputConfig:
    """Parse output configuration from dict."""
    verbosity_str = data.get("verbosity", "normal")
    try:
        verbosity = OutputVerbosity(verbosity_str)
    except ValueError:
        verbosity = OutputVerbosity.NORMAL

    return OutputConfig(
        verbosity=verbosity,
        show_timestamps=data.get("show_timestamps", True),
        color_enabled=data.get("color_enabled", True),
        show_file_changes=data.get("show_file_changes", True),
        show_thinking=data.get("show_thinking", True),
    )


def _parse_knowledge_config(data: dict[str, Any]) -> KnowledgeConfig:
    """Parse knowledge configuration from dict."""
    return KnowledgeConfig(
        embedding_provider=data.get("embedding_provider", "local"),
        local_model=data.get("local_model", "nomic-embed-text"),
        auto_index=data.get("auto_index", True),
    )


def _parse_data_config(data: dict[str, Any]) -> DataConfig:
    """Parse data configuration from dict."""
    return DataConfig(
        directory=data.get("directory", "~/.perpetualcc/data"),
        checkpoint_retention_days=data.get("checkpoint_retention_days", 30),
        episode_retention_days=data.get("episode_retention_days", 90),
    )


def load_config(config_path: Path | None = None) -> PerpetualCCConfig:
    """Load configuration from TOML file.

    Args:
        config_path: Path to config file. Uses default if not provided.

    Returns:
        Loaded configuration, or default if file doesn't exist.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        return PerpetualCCConfig.default()

    if tomllib is None:
        import warnings

        warnings.warn(
            "TOML parsing not available. Install tomli for Python 3.10: pip install tomli"
        )
        return PerpetualCCConfig.default()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        # Log warning but return default config
        import warnings

        warnings.warn(f"Failed to load config from {path}: {e}")
        return PerpetualCCConfig.default()

    return PerpetualCCConfig(
        brain=_parse_brain_config(data.get("brain", {})),
        sessions=_parse_sessions_config(data.get("sessions", {})),
        permissions=_parse_permissions_config(data.get("permissions", {})),
        notifications=_parse_notifications_config(data.get("notifications", {})),
        output=_parse_output_config(data.get("output", {})),
        knowledge=_parse_knowledge_config(data.get("knowledge", {})),
        data=_parse_data_config(data.get("data", {})),
    )


def _format_toml_value(value: Any) -> str:
    """Format a Python value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # Escape and quote strings
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        items = [_format_toml_value(item) for item in value]
        return "[" + ", ".join(items) + "]"
    elif isinstance(value, Enum):
        return f'"{value.value}"'
    else:
        return f'"{value}"'


def save_config(config: PerpetualCCConfig, config_path: Path | None = None) -> None:
    """Save configuration to TOML file.

    Args:
        config: Configuration to save
        config_path: Path to config file. Uses default if not provided.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PerpetualCC Configuration",
        "# Generated by pcc config command",
        "# Edit this file to customize behavior",
        "",
        "[brain]",
        f"type = {_format_toml_value(config.brain.type)}",
        f"confidence_threshold = {config.brain.confidence_threshold}",
        "",
        "[brain.gemini]",
    ]

    if config.brain.gemini.api_key:
        lines.append(f"api_key = {_format_toml_value(config.brain.gemini.api_key)}")
    else:
        lines.append('# api_key = "env:GEMINI_API_KEY"  # or your API key directly')

    lines.extend(
        [
            f"model = {_format_toml_value(config.brain.gemini.model)}",
            "",
            "[brain.ollama]",
            f"host = {_format_toml_value(config.brain.ollama.host)}",
            f"model = {_format_toml_value(config.brain.ollama.model)}",
            f"timeout = {config.brain.ollama.timeout}",
            "",
            "[sessions]",
            f"max_concurrent = {config.sessions.max_concurrent}",
            f"auto_resume = {_format_toml_value(config.sessions.auto_resume)}",
            f"persist_history = {_format_toml_value(config.sessions.persist_history)}",
            f"checkpoint_retention_days = {config.sessions.checkpoint_retention_days}",
            "",
            "[permissions]",
            f"auto_approve_low_risk = {_format_toml_value(config.permissions.auto_approve_low_risk)}",
            f"safe_directories = {_format_toml_value(config.permissions.safe_directories)}",
            f"blocked_commands = {_format_toml_value(config.permissions.blocked_commands)}",
            "",
            "[notifications]",
            f"enabled = {_format_toml_value(config.notifications.enabled)}",
            f"sound = {_format_toml_value(config.notifications.sound)}",
            f"on_question = {_format_toml_value(config.notifications.on_question)}",
            f"on_complete = {_format_toml_value(config.notifications.on_complete)}",
            f"on_rate_limit = {_format_toml_value(config.notifications.on_rate_limit)}",
            f"on_error = {_format_toml_value(config.notifications.on_error)}",
            "",
            "[output]",
            f"verbosity = {_format_toml_value(config.output.verbosity)}",
            f"show_timestamps = {_format_toml_value(config.output.show_timestamps)}",
            f"color_enabled = {_format_toml_value(config.output.color_enabled)}",
            f"show_file_changes = {_format_toml_value(config.output.show_file_changes)}",
            f"show_thinking = {_format_toml_value(config.output.show_thinking)}",
            "",
            "[knowledge]",
            f"embedding_provider = {_format_toml_value(config.knowledge.embedding_provider)}",
            f"local_model = {_format_toml_value(config.knowledge.local_model)}",
            f"auto_index = {_format_toml_value(config.knowledge.auto_index)}",
            "",
            "[data]",
            f"directory = {_format_toml_value(config.data.directory)}",
            f"checkpoint_retention_days = {config.data.checkpoint_retention_days}",
            f"episode_retention_days = {config.data.episode_retention_days}",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_default_config() -> str:
    """Generate default configuration as TOML string.

    Returns:
        Default configuration in TOML format
    """
    return """# PerpetualCC Configuration
# Copy this file to ~/.perpetualcc/config.toml and customize

[brain]
# Brain type: "rule_based" (default), "gemini", or "ollama"
type = "rule_based"

# Minimum confidence for auto-answering questions (0.0 - 1.0)
confidence_threshold = 0.7

[brain.gemini]
# Gemini API key - use "env:VAR_NAME" to read from environment
# api_key = "env:GEMINI_API_KEY"
model = "gemini-2.0-flash"

[brain.ollama]
# Ollama server address
host = "http://localhost:11434"

# Model to use (must be pulled first: ollama pull <model>)
model = "llama3.2"

# Request timeout in seconds
timeout = 120

[sessions]
# Maximum concurrent sessions
max_concurrent = 5

# Auto-resume after rate limits
auto_resume = true

# Save session history
persist_history = true

# Days to keep checkpoints
checkpoint_retention_days = 30

[permissions]
# Auto-approve low-risk operations (read, safe writes)
auto_approve_low_risk = true

# Directories considered safe for writes
safe_directories = ["src/", "tests/", "lib/", "app/", "packages/"]

# Commands that are always blocked
blocked_commands = ["rm -rf /", "sudo", "curl | sh"]

[notifications]
# Enable macOS notifications
enabled = true

# Play sound with notifications
sound = true

# Notify on specific events
on_question = true
on_complete = true
on_rate_limit = true
on_error = true

[output]
# Output verbosity: "quiet", "normal", or "verbose"
verbosity = "normal"

# Show timestamps on output lines
show_timestamps = true

# Enable colored output
color_enabled = true

# Show file change details
show_file_changes = true

# Show Claude's thinking process
show_thinking = true

[knowledge]
# Embedding provider: "local" (ollama), "gemini", or "hybrid"
embedding_provider = "local"

# Local embedding model (requires: ollama pull nomic-embed-text)
local_model = "nomic-embed-text"

# Auto-index project on session start
auto_index = true

[data]
# Data directory for sessions, checkpoints, and memory
directory = "~/.perpetualcc/data"

# Days to keep checkpoints
checkpoint_retention_days = 30

# Days to keep episode memory
episode_retention_days = 90
"""


# Global config instance (lazy loaded)
_config: PerpetualCCConfig | None = None


def get_config() -> PerpetualCCConfig:
    """Get the global configuration instance.

    Loads from file on first call, caches thereafter.

    Returns:
        The global configuration
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> PerpetualCCConfig:
    """Reload configuration from file.

    Returns:
        The reloaded configuration
    """
    global _config
    _config = load_config()
    return _config


def set_config(config: PerpetualCCConfig) -> None:
    """Set the global configuration instance.

    Useful for testing or programmatic configuration.

    Args:
        config: Configuration to set
    """
    global _config
    _config = config
