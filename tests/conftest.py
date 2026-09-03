import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Give every test its own data directory, so nothing touches the real one."""
    monkeypatch.setenv("COMPANIONAI_HOME", str(tmp_path))

    from companionai import paths

    importlib.reload(paths)
    for module_name in ("config", "net", "catalog", "character"):
        module = importlib.import_module(f"companionai.{module_name}")
        importlib.reload(module)

    import companionai.config as config
    import companionai.session as session

    config._settings = None
    # The session holds the active character and would otherwise carry a
    # companion from a previous test into this test's empty data directory.
    session._session = None
    paths.ensure_dirs()
    yield tmp_path
    session._session = None
