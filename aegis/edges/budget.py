from aegis.models.state import AegisState


def budget_gate_routing(state: AegisState) -> str:
    return "budget_gate"


def budget_decision_routing(state: AegisState) -> str:
    if state.human_decision == "approve":
        return "generate_code"
    elif state.human_decision == "optimize":
        return "remediate_spec"
    elif state.human_decision == "cancel":
        return "__end__"
    else:
        return "__end__"
