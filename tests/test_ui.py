"""Tests for the Streamlit UI module existence and core import sanity."""

import os

import pytest


def test_ui_file_exists():
    """The Streamlit app file must exist at ui/streamlit_app.py."""
    assert os.path.exists(os.path.join("ui", "streamlit_app.py"))


def test_ui_imports_work():
    """Core dependencies used by the UI must be importable."""
    from aegis.models.state import AegisState
    from aegis.graph import build_aegis_graph

    assert AegisState is not None
    assert build_aegis_graph is not None
