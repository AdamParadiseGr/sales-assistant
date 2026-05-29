"""
Shared pytest configuration.

Adds the project root to sys.path so tests can import `app.*` directly
without installing the package. Also configures pytest-asyncio in auto mode
so async tests don't need the @pytest.mark.asyncio decorator.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
