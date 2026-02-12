from aegis.models.state import AegisState


def retry_routing(state: AegisState) -> str:
    if state.retry_count < state.max_retries:
        return "remediate_spec"
    else:
        return "write_report"
