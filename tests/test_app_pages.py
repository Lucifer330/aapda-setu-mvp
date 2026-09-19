"""Smoke-test Streamlit pages with AppTest (no browser)."""

from pathlib import Path

import pytest

try:
    import matplotlib  # noqa: F401
except ImportError:
    pytest.skip("matplotlib not installed in this interpreter", allow_module_level=True)

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _at() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=45)
    at.run()
    assert not at.exception, at.exception
    return at


def test_default_dashboard_and_pages():
    at = _at()
    pages = ["Dashboard", "Incidents", "Vessels", "Analytics", "Evidence", "About"]
    for name in pages:
        btn = [b for b in at.button if b.label == name]
        assert btn, f"missing nav {name}"
        btn[0].click().run()
        assert not at.exception, f"{name}: {at.exception}"


def test_vessel_select_does_not_crash():
    at = _at()
    [b for b in at.button if b.label == "Vessels"][0].click().run()
    assert not at.exception
    box = at.selectbox[0]
    options = list(box.options)
    for opt in options[:3]:
        box.set_value(opt).run()
        assert not at.exception, opt
