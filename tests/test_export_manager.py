"""Unit tests for export manager."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import json
import csv
import os
from pathlib import Path
from perpetualcc.ui.web.export_manager import SessionExporter


class TestSessionExporter:
    """Test suite for SessionExporter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = "/tmp/test_exports"
        self.exporter = SessionExporter(self.test_dir)
        self.session_id = "test-session-001"
        self.session_data = {
            "id": self.session_id,
            "status": "completed",
            "logs": [{"timestamp": "2024-01-01", "message": "test"}]
        }
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_export_session_json(self):
        """Test JSON export."""
        filepath = self.exporter.export_session_json(
            self.session_id, 
            self.session_data
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
            assert data["session_id"] == self.session_id
            assert "exported_at" in data
            assert data["data"] == self.session_data
    
    def test_export_session_csv(self):
        """Test CSV export."""
        logs = [
            {"timestamp": "2024-01-01", "level": "INFO", "message": "test1"},
            {"timestamp": "2024-01-02", "level": "ERROR", "message": "test2"},
        ]
        
        filepath = self.exporter.export_session_csv(self.session_id, logs)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify content
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["message"] == "test1"
            assert rows[1]["level"] == "ERROR"
    
    def test_export_session_csv_empty(self):
        """Test CSV export with empty logs."""
        filepath = self.exporter.export_session_csv(self.session_id, [])
        
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "timestamp" in headers
    
    def test_export_config(self):
        """Test config export."""
        config = {"brain": "kimi", "risk": "high"}
        
        filepath = self.exporter.export_config(self.session_id, config)
        
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            assert data["session_id"] == self.session_id
            assert data["config"] == config
    
    def test_list_exports(self):
        """Test listing exports."""
        # Create some exports
        self.exporter.export_session_json(self.session_id, self.session_data)
        self.exporter.export_config(self.session_id, {})
        
        exports = self.exporter.list_exports()
        
        assert len(exports) == 2
        assert all("filename" in e for e in exports)
        assert all("size" in e for e in exports)
        assert all("created" in e for e in exports)
    
    def test_list_exports_empty(self):
        """Test listing exports when none exist."""
        exports = self.exporter.list_exports()
        assert exports == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
