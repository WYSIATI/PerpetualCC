"""Analytics module for PerpetualCC Web UI."""

from datetime import datetime
from typing import Any

class SessionAnalytics:
    """Track and analyze session metrics."""
    
    def __init__(self):
        self.metrics: dict[str, dict[str, Any]] = {}
    
    def record_session_start(self, session_id: str, config: dict) -> None:
        """Record session start."""
        self.metrics[session_id] = {
            "start_time": datetime.now().isoformat(),
            "config": config,
            "events": [],
            "status": "running"
        }
    
    def record_event(self, session_id: str, event_type: str, data: dict) -> None:
        """Record session event."""
        if session_id in self.metrics:
            self.metrics[session_id]["events"].append({
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "data": data
            })
    
    def record_session_end(self, session_id: str, status: str = "completed") -> None:
        """Record session end."""
        if session_id in self.metrics:
            start = datetime.fromisoformat(self.metrics[session_id]["start_time"])
            duration = (datetime.now() - start).total_seconds()
            
            self.metrics[session_id].update({
                "end_time": datetime.now().isoformat(),
                "duration_seconds": duration,
                "status": status
            })
    
    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session."""
        if session_id not in self.metrics:
            return {}
        
        metric = self.metrics[session_id]
        events = metric.get("events", [])
        
        return {
            "session_id": session_id,
            "duration_seconds": metric.get("duration_seconds", 0),
            "status": metric.get("status", "unknown"),
            "event_count": len(events),
            "event_breakdown": self._count_events_by_type(events),
            "start_time": metric.get("start_time"),
            "end_time": metric.get("end_time")
        }
    
    def get_overall_stats(self) -> dict[str, Any]:
        """Get overall statistics."""
        if not self.metrics:
            return {
                "total_sessions": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "success_rate": 0
            }
        
        completed = [m for m in self.metrics.values() if m.get("status") == "completed"]
        durations = [m.get("duration_seconds", 0) for m in self.metrics.values() if "duration_seconds" in m]
        
        return {
            "total_sessions": len(self.metrics),
            "completed_sessions": len(completed),
            "total_duration": sum(durations),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "success_rate": (len(completed) / len(self.metrics) * 100) if self.metrics else 0
        }
    
    def _count_events_by_type(self, events: list) -> dict[str, int]:
        """Count events by type."""
        counts = {}
        for event in events:
            event_type = event.get("type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts


# Global analytics instance
analytics = SessionAnalytics()
