"""Simple test runner without pytest dependency."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from datetime import datetime

from perpetualcc.ui.web.analytics import SessionAnalytics
from perpetualcc.ui.web.export_manager import SessionExporter


class TestAnalytics(unittest.TestCase):
    """Test analytics module."""
    
    def setUp(self):
        self.analytics = SessionAnalytics()
        self.session_id = "test-001"
        self.config = {"brain": "kimi"}
    
    def test_record_session_start(self):
        self.analytics.record_session_start(self.session_id, self.config)
        self.assertIn(self.session_id, self.analytics.metrics)
        self.assertEqual(self.analytics.metrics[self.session_id]["config"], self.config)
    
    def test_record_event(self):
        self.analytics.record_session_start(self.session_id, self.config)
        self.analytics.record_event(self.session_id, "log", {"msg": "test"})
        events = self.analytics.metrics[self.session_id]["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "log")
    
    def test_overall_stats_empty(self):
        stats = self.analytics.get_overall_stats()
        self.assertEqual(stats["total_sessions"], 0)


class TestExportManager(unittest.TestCase):
    """Test export manager."""
    
    def setUp(self):
        self.exporter = SessionExporter("/tmp/test_exports_simple")
        self.session_id = "test-002"
        self.data = {"id": self.session_id, "status": "completed"}
    
    def tearDown(self):
        import shutil
        if os.path.exists("/tmp/test_exports_simple"):
            shutil.rmtree("/tmp/test_exports_simple")
    
    def test_export_json(self):
        filepath = self.exporter.export_session_json(self.session_id, self.data)
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.json'))


class TestWebApp(unittest.TestCase):
    """Test web app basic functionality."""
    
    def test_imports(self):
        """Test that app modules can be imported."""
        try:
            from perpetualcc.ui.web import app
            from perpetualcc.ui.web import analytics
            from perpetualcc.ui.web import export_manager
            success = True
        except ImportError as e:
            success = False
            print(f"Import error: {e}")
        
        self.assertTrue(success)


def run_tests():
    """Run all tests."""
    print("🧪 Running PerpetualCC Web UI Tests")
    print("=" * 50)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAnalytics))
    suite.addTests(loader.loadTestsFromTestCase(TestExportManager))
    suite.addTests(loader.loadTestsFromTestCase(TestWebApp))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
