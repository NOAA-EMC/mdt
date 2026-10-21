"""Explicit opt-in policy for tests that require external services."""

import pytest

pytest_plugins = ["pytester"]

_EXTERNAL_TESTS = {
    "network": "--run-network",
    "ecflow": "--run-ecflow",
    "hpc": "--run-hpc",
}


def pytest_addoption(parser):
    """Expose independent switches for each external integration requirement."""
    for marker, option in _EXTERNAL_TESTS.items():
        parser.addoption(option, action="store_true", default=False, help=f"Run tests requiring {marker} integration resources.")


def pytest_collection_modifyitems(config, items):
    """Keep ordinary pytest runs offline and independent of external services."""
    for item in items:
        for marker, option in _EXTERNAL_TESTS.items():
            if item.get_closest_marker(marker) is not None and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"Requires explicit {option}; see the integration testing guide."))
