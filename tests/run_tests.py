"""Test runner for PerpetualCC Web UI tests."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all unit tests."""
    test_dir = Path(__file__).parent
    
    print("🧪 Running PerpetualCC Web UI Tests")
    print("=" * 50)
    print()
    
    test_files = [
        "test_analytics.py",
        "test_export_manager.py",
        "test_web_app.py",
    ]
    
    results = []
    
    for test_file in test_files:
        print(f"Running {test_file}...")
        test_path = test_dir / test_file
        
        if test_path.exists():
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✅ {test_file} PASSED")
                results.append((test_file, True, "Passed"))
            else:
                print(f"  ❌ {test_file} FAILED")
                print(result.stdout)
                results.append((test_file, False, "Failed"))
        else:
            print(f"  ⚠️  {test_file} NOT FOUND")
            results.append((test_file, False, "Not found"))
        
        print()
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_file, success, status in results:
        icon = "✅" if success else "❌"
        print(f"{icon} {test_file}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
