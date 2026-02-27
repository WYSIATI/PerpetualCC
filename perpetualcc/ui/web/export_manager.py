"""Export functionality for PerpetualCC."""

import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

# Session ID validation pattern: alphanumeric, hyphens, underscores only
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
SESSION_ID_MAX_LENGTH = 128


class ExportError(Exception):
    """Exception raised for export-related errors."""

    pass


class SessionExporter:
    """Export session data in various formats."""

    def __init__(self, export_dir: str = "./exports"):
        self.export_dir = Path(export_dir).resolve()
        self._ensure_export_dir()

    def _ensure_export_dir(self) -> None:
        """Ensure export directory exists."""
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Export directory ready: {self.export_dir}")
        except OSError as e:
            logger.error(f"Failed to create export directory: {e}")
            raise ExportError(f"Failed to create export directory: {e}")

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID to prevent path traversal and other attacks.

        Args:
            session_id: The session ID to validate.

        Raises:
            ExportError: If the session ID is invalid.
        """
        if not session_id:
            raise ExportError("Session ID is required")

        if len(session_id) > SESSION_ID_MAX_LENGTH:
            raise ExportError(f"Session ID must be {SESSION_ID_MAX_LENGTH} characters or less")

        if not SESSION_ID_PATTERN.match(session_id):
            raise ExportError(
                "Session ID must contain only alphanumeric characters, hyphens, and underscores"
            )

        # Additional path traversal checks
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            logger.warning(f"Potential path traversal attempt in session_id: {session_id[:50]}")
            raise ExportError("Invalid session ID")

    def _safe_filepath(self, filename: str) -> Path:
        """Create a safe filepath within the export directory.

        Args:
            filename: The filename to use.

        Returns:
            Resolved Path within export directory.

        Raises:
            ExportError: If the resulting path would be outside export directory.
        """
        filepath = (self.export_dir / filename).resolve()

        # Ensure the resolved path is within export_dir
        try:
            filepath.relative_to(self.export_dir)
        except ValueError:
            logger.warning(f"Path traversal attempt detected: {filename}")
            raise ExportError("Invalid filename")

        return filepath

    def export_session_json(self, session_id: str, session_data: dict) -> str:
        """Export session as JSON.

        Args:
            session_id: The session identifier.
            session_data: The session data to export.

        Returns:
            Path to the exported file.

        Raises:
            ExportError: If export fails.
        """
        self._validate_session_id(session_id)
        logger.info(f"Exporting session {session_id} as JSON")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{timestamp}.json"
        filepath = self._safe_filepath(filename)

        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "data": session_data,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Successfully exported JSON to: {filepath}")
            return str(filepath)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write JSON export for session {session_id}: {e}")
            raise ExportError(f"Failed to write export file: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize session data for {session_id}: {e}")
            raise ExportError(f"Failed to serialize session data: {e}")

    def export_session_csv(self, session_id: str, logs: list[dict]) -> str:
        """Export session logs as CSV.

        Args:
            session_id: The session identifier.
            logs: List of log entries to export.

        Returns:
            Path to the exported file.

        Raises:
            ExportError: If export fails.
        """
        self._validate_session_id(session_id)
        logger.info(f"Exporting session {session_id} logs as CSV ({len(logs)} entries)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_logs_{timestamp}.csv"
        filepath = self._safe_filepath(filename)

        try:
            if not logs:
                # Create empty CSV with headers
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "level", "message"])
                logger.info(f"Created empty CSV at: {filepath}")
                return str(filepath)

            # Get all possible fields
            fieldnames = set()
            for log in logs:
                if isinstance(log, dict):
                    fieldnames.update(log.keys())
            fieldnames = sorted(fieldnames)

            if not fieldnames:
                fieldnames = ["timestamp", "level", "message"]

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for log in logs:
                    if isinstance(log, dict):
                        writer.writerow(log)

            logger.info(f"Successfully exported CSV to: {filepath}")
            return str(filepath)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write CSV export for session {session_id}: {e}")
            raise ExportError(f"Failed to write export file: {e}")

    def export_config(self, session_id: str, config: dict) -> str:
        """Export session configuration.

        Args:
            session_id: The session identifier.
            config: The configuration to export.

        Returns:
            Path to the exported file.

        Raises:
            ExportError: If export fails.
        """
        self._validate_session_id(session_id)
        logger.info(f"Exporting config for session {session_id}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_config_{timestamp}.json"
        filepath = self._safe_filepath(filename)

        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "config": config,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Successfully exported config to: {filepath}")
            return str(filepath)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write config export for session {session_id}: {e}")
            raise ExportError(f"Failed to write export file: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize config for {session_id}: {e}")
            raise ExportError(f"Failed to serialize config: {e}")

    def list_exports(self) -> list[dict]:
        """List all exported files.

        Returns:
            List of export file information dictionaries.
        """
        exports = []

        try:
            if not self.export_dir.exists():
                logger.warning(f"Export directory does not exist: {self.export_dir}")
                return exports

            for filepath in self.export_dir.glob("*"):
                if filepath.is_file():
                    try:
                        stat = filepath.stat()
                        exports.append(
                            {
                                "filename": filepath.name,
                                "path": str(filepath),
                                "size": stat.st_size,
                                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            }
                        )
                    except OSError as e:
                        logger.warning(f"Failed to stat file {filepath}: {e}")
                        continue

            logger.debug(f"Listed {len(exports)} export files")
            return sorted(exports, key=lambda x: x["created"], reverse=True)
        except OSError as e:
            logger.error(f"Failed to list exports: {e}")
            return exports

    def cleanup_old_exports(self, max_age_hours: int = 24) -> int:
        """Remove export files older than specified age.

        Args:
            max_age_hours: Maximum age of files to keep in hours.

        Returns:
            Number of files removed.
        """
        if max_age_hours < 1:
            raise ValueError("max_age_hours must be at least 1")

        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        removed = 0

        logger.info(f"Cleaning up exports older than {max_age_hours} hours")

        try:
            if not self.export_dir.exists():
                return 0

            for filepath in self.export_dir.glob("*"):
                if filepath.is_file():
                    try:
                        file_age = current_time - filepath.stat().st_mtime
                        if file_age > max_age_seconds:
                            filepath.unlink()
                            removed += 1
                            logger.debug(f"Removed old export: {filepath.name}")
                    except OSError as e:
                        logger.warning(f"Failed to remove old export {filepath}: {e}")
                        continue

            logger.info(f"Cleanup complete: removed {removed} old export files")
            return removed
        except OSError as e:
            logger.error(f"Failed during export cleanup: {e}")
            return removed

    def delete_export(self, filename: str) -> bool:
        """Delete a specific export file.

        Args:
            filename: Name of the file to delete.

        Returns:
            True if file was deleted, False otherwise.

        Raises:
            ExportError: If deletion fails or filename is invalid.
        """
        # Validate filename (prevent path traversal)
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            raise ExportError("Invalid filename")

        filepath = self._safe_filepath(filename)

        try:
            if not filepath.exists():
                logger.warning(f"Export file not found: {filename}")
                return False

            filepath.unlink()
            logger.info(f"Deleted export file: {filename}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete export {filename}: {e}")
            raise ExportError(f"Failed to delete export file: {e}")


# Global exporter instance
exporter = SessionExporter()
