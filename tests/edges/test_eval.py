import pytest
from aegis.models.state import AegisState
from aegis.edges.eval import eval_gate_routing


def test_eval_gate_routing_passed():
    assert eval_gate_routing(AegisState(eval_result={"passed": True})) == "write_report"


def test_eval_gate_routing_failed():
    assert (
        eval_gate_routing(
            AegisState(eval_result={"passed": False, "failures": ["Loss too high"]})
        )
        == "remediate_spec"
    )


def test_eval_gate_routing_no_result():
    assert eval_gate_routing(AegisState(eval_result=None)) == "write_report"
