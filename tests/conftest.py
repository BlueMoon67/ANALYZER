"""
conftest.py — pytest configuration for Store Intelligence tests.

1. Adds pipeline/ to sys.path so test_pipeline.py can import emit, tracker, etc.
2. Adds project root to sys.path so 'app' package is importable.
3. Resets the database engine before each test module so DATABASE_URL env overrides work.
"""
import sys
import os
from pathlib import Path

# Add pipeline/ to path for pipeline module imports (emit, tracker, etc.)
PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Add project root to path for 'app' package imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Reset the lazy engine so DATABASE_URL env vars set before import take effect."""
    try:
        import app.database as db
        db.reset_engine()
    except ImportError:
        pass  # app not importable yet — that's fine
