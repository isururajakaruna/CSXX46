#!/usr/bin/env python3
"""
Test script to verify transitions directory setup
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_transitions_dir():
    """Test transitions directory setup"""
    # Test 1: Local transitions directory
    local_dir = Path(__file__).parent / "transitions"
    print(f"Local transitions dir: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Test write permission
    test_file = local_dir / "test_write.txt"
    try:
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        print("[PASS] Local directory is writable")
    except Exception as e:
        print(f"[FAIL] Local directory write failed: {e}")

    # Test 2: Absolute path
    abs_dir = local_dir.absolute()
    print(f"Absolute path: {abs_dir}")

    # Test 3: Path with spaces and special chars (simulate real config)
    config_transitions_dir = str(abs_dir)
    if config_transitions_dir:
        resolved_dir = Path(config_transitions_dir.strip())
        print(f"Resolved from config: {resolved_dir}")
        resolved_dir.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file2 = resolved_dir / "config_test.txt"
        try:
            with open(test_file2, "w") as f:
                f.write("config test")
            test_file2.unlink()
            print("[PASS] Config directory is writable")
        except Exception as e:
            print(f"[FAIL] Config directory write failed: {e}")


if __name__ == "__main__":
    test_transitions_dir()
