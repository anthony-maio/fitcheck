from aegis.models.state import AegisState


def eval_gate_routing(state: AegisState) -> str:
    if state.eval_result is None:
        return "write_report"
    if state.eval_result.get("passed", False):
        return "write_report"
    else:
        return "remediate_spec"
