import sqlite3

import pytest


@pytest.fixture
def in_memory_db():
    """Provide an in-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
