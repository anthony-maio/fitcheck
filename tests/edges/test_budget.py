import pytest
from aegis.models.state import AegisState, CostEstimate, BudgetPolicy
from aegis.edges.budget import budget_gate_routing, budget_decision_routing


def test_budget_gate_routing_no_estimate():
    state = AegisState(cost_estimate=None)
    assert budget_gate_routing(state) == "budget_gate"


def test_budget_gate_routing_exceeds_max():
    state = AegisState(
        cost_estimate=CostEstimate(
            estimated_cost_usd=10.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=10.0,
            cost_breakdown={},
        ),
        budget_policy=BudgetPolicy(max_budget_usd=5.0),
    )
    assert budget_gate_routing(state) == "budget_gate"


def test_budget_gate_routing_auto_approve():
    state = AegisState(
        cost_estimate=CostEstimate(
            estimated_cost_usd=1.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={},
        ),
        budget_policy=BudgetPolicy(soft_threshold_usd=2.0),
    )
    assert budget_gate_routing(state) == "budget_gate"


def test_budget_decision_routing_approve():
    assert budget_decision_routing(AegisState(human_decision="approve")) == "generate_code"


def test_budget_decision_routing_optimize():
    assert budget_decision_routing(AegisState(human_decision="optimize")) == "remediate_spec"


def test_budget_decision_routing_cancel():
    assert budget_decision_routing(AegisState(human_decision="cancel")) == "__end__"
