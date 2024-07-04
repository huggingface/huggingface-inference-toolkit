import os

import pytest


@pytest.fixture(scope = "session")
def cache_test_dir():
    yield os.environ.get("CACHE_TEST_DIR", "./tests")
