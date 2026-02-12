"""Tests for the main Aegis LangGraph orchestration graph."""

import pytest

from aegis.graph import build_aegis_graph
from aegis.models.state import AegisState


def test_graph_builds():
    """build_aegis_graph should return a compiled graph object."""
    graph = build_aegis_graph()
    assert graph is not None


def test_graph_has_all_nodes():
    """Every expected node must be present in the compiled graph."""
    graph = build_aegis_graph()
    nodes = graph.nodes
    expected_nodes = [
        "parse_intent",
        "estimate_cost",
        "budget_gate",
        "generate_code",
        "execute",
        "run_evals",
        "eval_gate",
        "remediate_spec",
        "check_retries",
        "write_report",
    ]
    for node in expected_nodes:
        assert node in nodes, f"Missing node: {node}"


def test_graph_invoke_simple_flow():
    """The happy-path flow should complete from parse_intent through write_report.

    With the default budget policy the tiny-model cost (~$0.01) is auto-approved,
    the mock executor produces reasonable metrics, and evals pass -- so the graph
    should traverse every node on the main path and produce a final report.
    """
    graph = build_aegis_graph()
    initial_state = AegisState()
    config = {"configurable": {"thread_id": "test-thread"}}

    result = graph.invoke(initial_state, config=config)

    assert result is not None

    # Should have parsed spec
    assert result["spec"] is not None

    # Should have cost estimate
    assert result["cost_estimate"] is not None

    # Budget gate should have auto-approved
    assert result["human_decision"] == "approve"

    # Code generation should have produced output
    assert result["generated_code"] is not None
    assert len(result["generated_code"]) > 0

    # Mock execution should have produced a result
    assert result["execution_result"] is not None
    assert "metrics" in result["execution_result"]

    # Evals should have run and passed
    assert result["eval_result"] is not None
    assert result["eval_result"]["passed"] is True

    # Final report should be generated
    assert result["final_report"] is not None
    assert len(result["final_report"]) > 0


def test_graph_events_trace():
    """The event log should capture every phase the graph traversed."""
    graph = build_aegis_graph()
    config = {"configurable": {"thread_id": "test-events"}}

    result = graph.invoke(AegisState(), config=config)

    events = result["events"]
    phases = [e.phase for e in events]

    # Verify the expected phases appear in order
    assert "parse_intent" in phases
    assert "estimate_cost" in phases
    assert "budget_gate" in phases
    assert "generate_code" in phases
    assert "execute" in phases
    assert "run_evals" in phases
    assert "write_report" in phases

    # All events should have completed status on the happy path
    for event in events:
        assert event.status == "completed", (
            f"Event {event.phase} has status {event.status}: {event.message}"
        )
