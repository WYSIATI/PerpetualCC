"""Unit tests for analytics module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from datetime import datetime, timedelta
from perpetualcc.ui.web.analytics import SessionAnalytics


class TestSessionAnalytics:
    """Test suite for SessionAnalytics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics = SessionAnalytics()
        self.session_id = "test-session-001"
        self.config = {"brain": "kimi", "risk": "high"}
    
    def test_record_session_start(self):
        """Test recording session start."""
        self.analytics.record_session_start(self.session_id, self.config)
        
        assert self.session_id in self.analytics.metrics
        assert self.analytics.metrics[self.session_id]["config"] == self.config
        assert "start_time" in self.analytics.metrics[self.session_id]
        assert self.analytics.metrics[self.session_id]["status"] == "running"
    
    def test_record_event(self):
        """Test recording events."""
        self.analytics.record_session_start(self.session_id, self.config)
        self.analytics.record_event(self.session_id, "log", {"message": "test"})
        
        events = self.analytics.metrics[self.session_id]["events"]
        assert len(events) == 1
        assert events[0]["type"] == "log"
        assert events[0]["data"]["message"] == "test"
    
    def test_record_session_end(self):
        """Test recording session end."""
        self.analytics.record_session_start(self.session_id, self.config)
        
        # Simulate time passing
        import time
        time.sleep(0.1)
        
        self.analytics.record_session_end(self.session_id, "completed")
        
        metric = self.analytics.metrics[self.session_id]
        assert metric["status"] == "completed"
        assert "end_time" in metric
        assert "duration_seconds" in metric
        assert metric["duration_seconds"] > 0
    
    def test_get_session_stats(self):
        """Test getting session statistics."""
        self.analytics.record_session_start(self.session_id, self.config)
        self.analytics.record_event(self.session_id, "log", {"message": "test"})
        self.analytics.record_session_end(self.session_id, "completed")
        
        stats = self.analytics.get_session_stats(self.session_id)
        
        assert stats["session_id"] == self.session_id
        assert stats["status"] == "completed"
        assert stats["event_count"] == 1
        assert "duration_seconds" in stats
    
    def test_get_session_stats_nonexistent(self):
        """Test getting stats for non-existent session."""
        stats = self.analytics.get_session_stats("nonexistent")
        assert stats == {}
    
    def test_get_overall_stats_empty(self):
        """Test overall stats with no sessions."""
        stats = self.analytics.get_overall_stats()
        
        assert stats["total_sessions"] == 0
        assert stats["total_duration"] == 0
        assert stats["avg_duration"] == 0
        assert stats["success_rate"] == 0
    
    def test_get_overall_stats_with_sessions(self):
        """Test overall stats with multiple sessions."""
        # Session 1 - completed
        self.analytics.record_session_start("session-1", {})
        self.analytics.record_session_end("session-1", "completed")
        
        # Session 2 - failed
        self.analytics.record_session_start("session-2", {})
        self.analytics.record_session_end("session-2", "failed")
        
        stats = self.analytics.get_overall_stats()
        
        assert stats["total_sessions"] == 2
        assert stats["completed_sessions"] == 1
        assert stats["success_rate"] == 50.0
    
    def test_event_breakdown(self):
        """Test event breakdown in stats."""
        self.analytics.record_session_start(self.session_id, self.config)
        self.analytics.record_event(self.session_id, "log", {})
        self.analytics.record_event(self.session_id, "log", {})
        self.analytics.record_event(self.session_id, "error", {})
        
        stats = self.analytics.get_session_stats(self.session_id)
        
        assert stats["event_breakdown"]["log"] == 2
        assert stats["event_breakdown"]["error"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
