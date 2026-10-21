"""Verify that external integrations require explicit pytest opt-ins."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("options", "passed", "skipped"),
    [((), 1, 4), (("--run-network",), 2, 3), (("--run-network", "--run-ecflow", "--run-hpc"), 5, 0)],
)
def test_integration_opt_ins(pytester, options, passed, skipped):
    """Default and partial opt-ins never execute tests with unmet requirements."""
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makeini("""[pytest]
markers =
    network: external data
    ecflow: real ecFlow
    hpc: real scheduler
""")
    pytester.makepyfile("""
import pytest


def test_offline():
    pass


@pytest.mark.network
def test_network():
    pass


@pytest.mark.ecflow
def test_ecflow():
    pass


@pytest.mark.hpc
def test_hpc():
    pass


@pytest.mark.network
@pytest.mark.ecflow
def test_combined():
    pass
""")
    result = pytester.runpytest_subprocess("-q", *options)
    result.assert_outcomes(passed=passed, skipped=skipped)
