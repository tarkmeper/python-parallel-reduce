"""
Add a configuration that enables skipping the benchrmark tests so that by default we aren't running them as they are
several orders of magnitude slower than the rest of the tests.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="Enable the benchrmark tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_runtest_setup(item):
    if 'benchmark' in item.keywords and not item.config.getoption("--benchmark"):
        pytest.skip("add --benchmark option to run this test")