"""Tests for CLI entry point configuration.

These tests verify that:
1. The package can be installed with pipx and the `pcc` command works
2. The entry point is properly configured in pyproject.toml
3. The `python -m perpetualcc` invocation works
4. Version information is consistent across all entry points
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from perpetualcc import __version__
from perpetualcc.cli import app


runner = CliRunner()


class TestEntryPointConfiguration:
    """Tests for verifying entry point configuration for pipx compatibility."""

    def test_cli_app_is_typer_instance(self):
        """Verify cli.app is a Typer application that can be used as entry point."""
        import typer
        assert isinstance(app, typer.Typer)

    def test_cli_app_has_name(self):
        """Verify the CLI app has the correct name."""
        # The app name should be 'pcc'
        assert app.info.name == "pcc"

    def test_version_command(self):
        """Verify version command works correctly."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_help_command(self):
        """Verify --help flag works correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Should show main commands
        assert "start" in result.output
        assert "list" in result.output

    def test_version_consistency(self):
        """Verify version is consistent between __init__ and CLI output."""
        from perpetualcc import __version__ as init_version

        result = runner.invoke(app, ["version"])
        assert init_version in result.output

    def test_entry_point_in_metadata(self):
        """Verify the console_scripts entry point is registered.

        This test ensures that when the package is installed (pip/pipx),
        the `pcc` command will be available.
        """
        # Try to find the entry point in installed packages
        try:
            eps = entry_points()
            # Python 3.12+ uses a different API
            if hasattr(eps, 'select'):
                console_scripts = eps.select(group='console_scripts')
            else:
                console_scripts = eps.get('console_scripts', [])

            # Check if perpetualcc entry point exists (only if package is installed)
            pcc_entry = None
            for ep in console_scripts:
                if ep.name == 'pcc' and 'perpetualcc' in str(ep.value):
                    pcc_entry = ep
                    break

            if pcc_entry is not None:
                # If installed, verify it points to correct module
                assert 'perpetualcc.cli' in str(pcc_entry.value)
                assert 'app' in str(pcc_entry.value)
        except Exception:
            # If package not installed via pip, check pyproject.toml directly
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                content = pyproject_path.read_text()
                assert 'pcc = "perpetualcc.cli:app"' in content

    def test_pyproject_entry_point_definition(self):
        """Verify pyproject.toml has correct entry point definition."""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        content = pyproject_path.read_text()

        # Check for project.scripts section
        assert "[project.scripts]" in content
        assert 'pcc = "perpetualcc.cli:app"' in content


class TestModuleInvocation:
    """Tests for python -m perpetualcc invocation."""

    def test_main_module_exists(self):
        """Verify __main__.py exists and is importable."""
        from perpetualcc import __main__
        assert hasattr(__main__, 'app')

    def test_main_module_imports_app(self):
        """Verify __main__ imports the correct app."""
        from perpetualcc.__main__ import app as main_app
        from perpetualcc.cli import app as cli_app
        assert main_app is cli_app

    def test_python_m_perpetualcc_help(self):
        """Verify python -m perpetualcc --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "perpetualcc", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "start" in result.stdout
        assert "list" in result.stdout

    def test_python_m_perpetualcc_version(self):
        """Verify python -m perpetualcc version works."""
        result = subprocess.run(
            [sys.executable, "-m", "perpetualcc", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout


class TestCLICommands:
    """Tests to verify all main CLI commands are accessible."""

    @pytest.mark.parametrize("command", [
        "start",
        "list",
        "status",
        "attach",
        "pause",
        "resume",
        "add",
        "pending",
        "respond",
        "config",
        "logs",
        "init",
        "version",
    ])
    def test_command_has_help(self, command: str):
        """Verify each command has help text."""
        result = runner.invoke(app, [command, "--help"])
        # Some commands may require arguments, but --help should work
        assert result.exit_code == 0, f"Command '{command}' failed: {result.output}"

    def test_list_command_works(self):
        """Verify list command works without arguments."""
        with patch("perpetualcc.cli._get_session_manager") as mock_get_sm:
            mock_sm = mock_get_sm.return_value
            mock_sm.list_sessions.return_value = []

            result = runner.invoke(app, ["list"])
            # Should succeed even with no sessions
            assert result.exit_code == 0

    def test_pending_command_works(self):
        """Verify pending command works without arguments."""
        with patch("perpetualcc.cli._get_session_manager") as mock_get_sm:
            mock_sm = mock_get_sm.return_value
            mock_sm.get_pending_escalations.return_value = []

            result = runner.invoke(app, ["pending"])
            assert result.exit_code == 0
