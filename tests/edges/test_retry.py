import pytest
from aegis.models.state import AegisState
from aegis.edges.retry import retry_routing


def test_retry_routing_has_retries():
    assert retry_routing(AegisState(retry_count=1, max_retries=3)) == "remediate_spec"


def test_retry_routing_exhausted():
    assert retry_routing(AegisState(retry_count=3, max_retries=3)) == "write_report"


def test_retry_routing_at_zero():
    assert retry_routing(AegisState(retry_count=0, max_retries=3)) == "remediate_spec"
