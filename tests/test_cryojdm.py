"""Tests for `cryojdm` module."""

from typing import Generator

import pytest

import cryojdm


@pytest.fixture
def version() -> Generator[str, None, None]:
    """Sample pytest fixture."""
    yield cryojdm.__version__


def test_version(version: str) -> None:
    """Sample pytest test function with the pytest fixture as an argument."""
    assert version == cryojdm.__version__
