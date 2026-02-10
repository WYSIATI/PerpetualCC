"""Unit tests for web app routes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from fastapi.testclient import TestClient
from perpetualcc.ui.web.app import app


class TestWebApp:
    """Test suite for web app routes."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_list_sessions_endpoint(self):
        """Test sessions list endpoint."""
        response = self.client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)
    
    def test_start_session_endpoint(self):
        """Test start session endpoint."""
        session_id = "test-session-001"
        response = self.client.post(f"/api/sessions/{session_id}/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["session_id"] == session_id
    
    def test_start_session_with_config(self):
        """Test start session with configuration."""
        session_id = "test-session-002"
        config = {"brain": "kimi", "risk": "high"}
        
        response = self.client.post(
            f"/api/sessions/{session_id}/start",
            json=config
        )
        
        assert response.status_code == 200
    
    def test_stop_session_endpoint(self):
        """Test stop session endpoint."""
        session_id = "test-session-003"
        
        # Start session first
        self.client.post(f"/api/sessions/{session_id}/start")
        
        # Stop session
        response = self.client.post(f"/api/sessions/{session_id}/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
    
    def test_analytics_overall_endpoint(self):
        """Test overall analytics endpoint."""
        response = self.client.get("/api/analytics/overall")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "success_rate" in data
    
    def test_analytics_session_endpoint(self):
        """Test session analytics endpoint."""
        session_id = "test-session-004"
        
        # Create a session
        self.client.post(f"/api/sessions/{session_id}/start")
        self.client.post(f"/api/sessions/{session_id}/stop")
        
        response = self.client.get(f"/api/analytics/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_exports_list_endpoint(self):
        """Test exports list endpoint."""
        response = self.client.get("/api/exports")
        
        assert response.status_code == 200
        data = response.json()
        assert "exports" in data
        assert isinstance(data["exports"], list)
    
    def test_export_json_endpoint(self):
        """Test JSON export endpoint."""
        session_id = "test-session-005"
        
        # Create session
        self.client.post(f"/api/sessions/{session_id}/start")
        
        response = self.client.post(f"/api/sessions/{session_id}/export/json")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "filepath" in data
    
    def test_export_csv_endpoint(self):
        """Test CSV export endpoint."""
        session_id = "test-session-006"
        
        # Create session
        self.client.post(f"/api/sessions/{session_id}/start")
        
        response = self.client.post(f"/api/sessions/{session_id}/export/csv")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_theme_endpoint(self):
        """Test theme setting endpoint."""
        response = self.client.get("/api/theme/dark")
        
        assert response.status_code == 200
        data = response.json()
        assert data["theme"] == "dark"
        assert data["status"] == "set"
    
    def test_analytics_dashboard_page(self):
        """Test analytics dashboard page."""
        response = self.client.get("/analytics")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
