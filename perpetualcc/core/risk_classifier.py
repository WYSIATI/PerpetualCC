"""Risk classification for tool use permissions in Claude Code sessions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RiskLevel(Enum):
    """Risk level for a tool use request."""

    LOW = "low"  # Auto-approve
    MEDIUM = "medium"  # Use brain for evaluation
    HIGH = "high"  # Escalate to human / block


@dataclass(frozen=True)
class RiskClassification:
    """Result of risk classification for a tool use request."""

    level: RiskLevel
    reason: str
    matched_pattern: str | None = None


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk classification."""

    safe_directories: tuple[str, ...] = (
        "src/",
        "tests/",
        "test/",
        "lib/",
        "app/",
        "packages/",
        "components/",
        "utils/",
        "helpers/",
    )

    safe_bash_commands: tuple[str, ...] = (
        "npm",
        "yarn",
        "pnpm",
        "pip",
        "pip3",
        "pytest",
        "cargo",
        "go",
        "make",
        "node",
        "python",
        "python3",
        "tsc",
        "eslint",
        "prettier",
        "jest",
        "vitest",
        "mocha",
        "ruff",
        "black",
        "mypy",
        "ls",
        "pwd",
        "echo",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "sort",
        "uniq",
        "diff",
        "which",
        "whoami",
        "date",
        "env",
    )

    dangerous_bash_patterns: tuple[str, ...] = (
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+~",
        r"rm\s+-rf\s+\$HOME",
        r"rm\s+-rf\s+\*",
        r"rm\s+-rf\s+\.\.",
        r"sudo\s+",
        r"chmod\s+-R\s+777",
        r"chmod\s+777\s+/",
        r">\s*/etc/",
        r">\s*/usr/",
        r">\s*/bin/",
        r"curl\s+.*\|\s*(ba)?sh",
        r"wget\s+.*\|\s*(ba)?sh",
        r"\|\s*(ba)?sh",
        r"mkfs\.",
        r"dd\s+if=",
        r":(){:|:&};:",  # Fork bomb
        r"--force\s+.*(-d|--delete)",
        r"git\s+push\s+.*--force",
        r"git\s+push\s+-f",
        r"git\s+reset\s+--hard\s+origin",
    )

    medium_risk_bash_patterns: tuple[str, ...] = (
        r"git\s+",
        r"curl\s+",
        r"wget\s+",
        r"ssh\s+",
        r"scp\s+",
        r"rsync\s+",
        r"docker\s+",
        r"kubectl\s+",
        r"aws\s+",
        r"gcloud\s+",
        r"az\s+",
        r"rm\s+",
        r"mv\s+.*\.\.",
    )

    high_risk_file_patterns: tuple[str, ...] = (
        r"\.env$",
        r"\.env\.",
        r"credentials",
        r"secrets",
        r"\.pem$",
        r"\.key$",
        r"\.crt$",
        r"\.p12$",
        r"id_rsa",
        r"id_ed25519",
        r"\.aws/",
        r"\.ssh/",
        r"\.gnupg/",
        r"password",
        r"token",
        r"apikey",
        r"api_key",
    )

    config_file_patterns: tuple[str, ...] = (
        r"package\.json$",
        r"tsconfig\.json$",
        r"pyproject\.toml$",
        r"setup\.py$",
        r"Cargo\.toml$",
        r"go\.mod$",
        r"\.eslintrc",
        r"\.prettierrc",
        r"webpack\.config",
        r"vite\.config",
        r"jest\.config",
        r"babel\.config",
        r"Dockerfile",
        r"docker-compose",
        r"\.gitlab-ci\.yml$",
        r"\.github/workflows/",
        r"Makefile$",
    )


class RiskClassifier:
    """Classifies risk level of tool use requests from Claude Code."""

    def __init__(
        self,
        project_path: str | Path,
        config: RiskConfig | None = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.config = config or RiskConfig()

    def classify(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> RiskClassification:
        """
        Classify the risk level of a tool use request.

        Args:
            tool_name: Name of the tool being used (Read, Write, Edit, Bash, etc.)
            tool_input: Input parameters for the tool

        Returns:
            RiskClassification with level, reason, and optionally matched pattern
        """
        classifier_method = getattr(self, f"_classify_{tool_name.lower()}", None)
        if classifier_method:
            return classifier_method(tool_input)

        # Default: unknown tools are medium risk
        return RiskClassification(
            level=RiskLevel.MEDIUM,
            reason=f"Unknown tool: {tool_name}",
        )

    def _classify_read(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Read tool is always low risk - reading doesn't modify anything."""
        return RiskClassification(
            level=RiskLevel.LOW,
            reason="Read operations are safe",
        )

    def _classify_glob(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Glob tool is always low risk - just file pattern matching."""
        return RiskClassification(
            level=RiskLevel.LOW,
            reason="Glob operations are safe",
        )

    def _classify_grep(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Grep tool is always low risk - just content searching."""
        return RiskClassification(
            level=RiskLevel.LOW,
            reason="Grep operations are safe",
        )

    def _classify_webfetch(self, tool_input: dict[str, Any]) -> RiskClassification:
        """WebFetch is low risk - just fetching web content."""
        return RiskClassification(
            level=RiskLevel.LOW,
            reason="WebFetch operations are safe",
        )

    def _classify_websearch(self, tool_input: dict[str, Any]) -> RiskClassification:
        """WebSearch is low risk - just searching."""
        return RiskClassification(
            level=RiskLevel.LOW,
            reason="WebSearch operations are safe",
        )

    def _classify_todowrite(self, tool_input: dict[str, Any]) -> RiskClassification:
        """TodoWrite is low risk - just internal task tracking."""
        return RiskClassification(
            level=RiskLevel.LOW,
            reason="TodoWrite operations are safe",
        )

    def _classify_task(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Task tool (subagents) is medium risk - depends on what it does."""
        return RiskClassification(
            level=RiskLevel.MEDIUM,
            reason="Task operations may perform various actions",
        )

    def _classify_write(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Classify Write tool based on file path."""
        file_path = tool_input.get("file_path", "")
        return self._classify_file_write(file_path)

    def _classify_edit(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Classify Edit tool based on file path."""
        file_path = tool_input.get("file_path", "")
        return self._classify_file_write(file_path)

    def _classify_file_write(self, file_path: str) -> RiskClassification:
        """Classify file write/edit operations."""
        # Check for high-risk file patterns (secrets, credentials, etc.)
        for pattern in self.config.high_risk_file_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return RiskClassification(
                    level=RiskLevel.HIGH,
                    reason=f"Writing to sensitive file: {file_path}",
                    matched_pattern=pattern,
                )

        # Check if file is outside project
        try:
            resolved_path = Path(file_path).resolve()
            if not str(resolved_path).startswith(str(self.project_path)):
                return RiskClassification(
                    level=RiskLevel.HIGH,
                    reason=f"Writing outside project directory: {file_path}",
                )
        except (OSError, ValueError):
            return RiskClassification(
                level=RiskLevel.MEDIUM,
                reason=f"Unable to resolve file path: {file_path}",
            )

        # Check for config file patterns (medium risk)
        for pattern in self.config.config_file_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return RiskClassification(
                    level=RiskLevel.MEDIUM,
                    reason=f"Modifying config file: {file_path}",
                    matched_pattern=pattern,
                )

        # Check if file is in safe directories
        relative_path = str(resolved_path.relative_to(self.project_path))
        for safe_dir in self.config.safe_directories:
            if relative_path.startswith(safe_dir):
                return RiskClassification(
                    level=RiskLevel.LOW,
                    reason=f"File in safe directory: {safe_dir}",
                )

        # Default: medium risk for other files
        return RiskClassification(
            level=RiskLevel.MEDIUM,
            reason=f"File not in known safe directory: {file_path}",
        )

    def _classify_bash(self, tool_input: dict[str, Any]) -> RiskClassification:
        """Classify Bash command based on command content."""
        command = tool_input.get("command", "")

        # Check for high-risk patterns first
        for pattern in self.config.dangerous_bash_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return RiskClassification(
                    level=RiskLevel.HIGH,
                    reason="Dangerous bash command pattern detected",
                    matched_pattern=pattern,
                )

        # Check for medium-risk patterns
        for pattern in self.config.medium_risk_bash_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return RiskClassification(
                    level=RiskLevel.MEDIUM,
                    reason="Network or system command detected",
                    matched_pattern=pattern,
                )

        # Extract the base command (first word)
        base_command = self._extract_base_command(command)

        # Check if it's a safe command
        if base_command in self.config.safe_bash_commands:
            return RiskClassification(
                level=RiskLevel.LOW,
                reason=f"Safe development command: {base_command}",
            )

        # Default: medium risk for unknown commands
        return RiskClassification(
            level=RiskLevel.MEDIUM,
            reason=f"Unknown bash command: {base_command}",
        )

    def _extract_base_command(self, command: str) -> str:
        """Extract the base command from a bash command string."""
        # Strip leading whitespace and handle common prefixes
        command = command.strip()

        # Handle environment variable assignments (VAR=value cmd)
        while "=" in command.split()[0] if command.split() else False:
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                command = parts[1]
            else:
                break

        # Handle command prefixes like 'time', 'env', etc.
        skip_prefixes = ("time", "env", "nohup", "nice")
        parts = command.split()
        while parts and parts[0] in skip_prefixes:
            parts = parts[1:]

        # Get the base command
        if parts:
            # Handle path prefixes (e.g., /usr/bin/npm)
            base = parts[0].split("/")[-1]
            return base

        return command


def classify_risk(
    tool_name: str,
    tool_input: dict[str, Any],
    project_path: str | Path,
    config: RiskConfig | None = None,
) -> RiskClassification:
    """
    Convenience function to classify risk without creating a classifier instance.

    Args:
        tool_name: Name of the tool being used
        tool_input: Input parameters for the tool
        project_path: Path to the project directory
        config: Optional risk configuration

    Returns:
        RiskClassification with level and reason
    """
    classifier = RiskClassifier(project_path, config)
    return classifier.classify(tool_name, tool_input)
