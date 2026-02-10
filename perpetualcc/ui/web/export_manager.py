"""Export functionality for PerpetualCC."""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionExporter:
    """Export session data in various formats."""
    
    def __init__(self, export_dir: str = "./exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_session_json(self, session_id: str, session_data: dict) -> str:
        """Export session as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{timestamp}.json"
        filepath = self.export_dir / filename
        
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "data": session_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)
    
    def export_session_csv(self, session_id: str, logs: list[dict]) -> str:
        """Export session logs as CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_logs_{timestamp}.csv"
        filepath = self.export_dir / filename
        
        if not logs:
            # Create empty CSV with headers
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "level", "message"])
            return str(filepath)
        
        # Get all possible fields
        fieldnames = set()
        for log in logs:
            fieldnames.update(log.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
        
        return str(filepath)
    
    def export_config(self, session_id: str, config: dict) -> str:
        """Export session configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_config_{timestamp}.json"
        filepath = self.export_dir / filename
        
        export_data = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "config": config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        return str(filepath)
    
    def list_exports(self) -> list[dict]:
        """List all exported files."""
        exports = []
        for filepath in self.export_dir.glob("*"):
            if filepath.is_file():
                exports.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "size": filepath.stat().st_size,
                    "created": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                })
        return sorted(exports, key=lambda x: x["created"], reverse=True)


# Global exporter instance
exporter = SessionExporter()
