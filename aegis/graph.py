"""Main LangGraph definition for the Aegis-ML fine-tuning orchestrator.

Wires every node and conditional edge into a single compiled ``StateGraph``
that can be invoked with an ``AegisState`` instance.  A ``MemorySaver``
checkpointer is attached so that each thread's state is persisted in memory
across ``invoke`` calls (useful for human-in-the-loop budget decisions).
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from aegis.models.state import AegisState, RunEvent
from aegis.nodes.intent import parse_intent_node
from aegis.nodes.intent_llm import parse_intent_llm
from aegis.nodes.profiler import estimate_cost_node
from aegis.nodes.gate import budget_gate_node
from aegis.nodes.generator import generate_code_node
from aegis.nodes.evaluator import run_evals_node
from aegis.nodes.remediate import remediate_spec_node
from aegis.nodes.reporter import write_report_node

from aegis.edges.budget import budget_gate_routing, budget_decision_routing
from aegis.edges.retry import retry_routing
from aegis.edges.eval import eval_gate_routing
from aegis.executors.modal_runner import ModalExecutor


# ---------------------------------------------------------------------------
# Stub / adapter nodes
# ---------------------------------------------------------------------------

def execute_node(state: AegisState) -> AegisState:
    """Execute training script via Modal (or mock fallback)."""
    if state.generated_code is None:
        event = RunEvent(
            phase="execute",
            status="failed",
            message="No code to execute",
        )
        return state.model_copy(update={"events": state.events + [event]})

    gpu = state.spec.target_gpu if state.spec else "a10g"
    executor = ModalExecutor(gpu=gpu)
    spec_json = state.spec.model_dump_json() if state.spec else "{}"
    raw = executor.execute(state.generated_code, spec_json)

    extracted_error = raw.get("extracted_error", "")

    result = {
        "metrics": raw.get("metrics", {}),
        "model_path": raw.get("model_path"),
        "duration_sec": raw.get("duration_sec"),
        "stdout": raw.get("stdout", ""),
        "stderr": raw.get("stderr", ""),
        "extracted_error": extracted_error,
    }

    if raw.get("returncode", -1) != 0:
        # Use extracted error (the actual traceback) for the event message
        err_summary = extracted_error or raw.get("stderr", "")
        # Show the last line of the error (the actual exception) in the summary
        err_lines = [l for l in err_summary.splitlines() if l.strip()]
        short_err = err_lines[-1][:200] if err_lines else "Unknown error"

        event = RunEvent(
            phase="execute",
            status="failed",
            message=f"Execution failed (rc={raw['returncode']}): {short_err}",
            data=result,
        )
        return state.model_copy(
            update={"execution_result": result, "events": state.events + [event]}
        )

    event = RunEvent(
        phase="execute",
        status="completed",
        message="Execution completed",
        data=result,
    )
    return state.model_copy(
        update={"execution_result": result, "events": state.events + [event]}
    )


def eval_gate_node(state: AegisState) -> AegisState:
    """Pass-through node for evaluation gate routing."""
    return state


def check_retries_node(state: AegisState) -> AegisState:
    """Pass-through node for retry checking."""
    return state


# ---------------------------------------------------------------------------
# Wrapper nodes (adapt multi-arg signatures for LangGraph)
# ---------------------------------------------------------------------------

def _wrap_parse_intent(state: AegisState) -> AegisState:
    """Wrap intent parser for LangGraph.  Prefers LLM, falls back to rules."""
    user_input = state.user_input or "Fine-tune tinyllama with LoRA"
    return parse_intent_llm(state, user_input=user_input)


def _wrap_remediate(state: AegisState) -> AegisState:
    """Wrap ``remediate_spec_node`` for LangGraph.

    Extracts a meaningful error message from the most recent evaluation or
    execution result so that the remediation heuristic can act on it.
    Eval failures are checked first since they are the most common trigger.
    Prefers ``extracted_error`` (the actual traceback) over raw stderr.
    """
    error_msg = "Generic training failure"
    if state.eval_result and state.eval_result.get("failures"):
        error_msg = "; ".join(state.eval_result["failures"])
    elif state.execution_result:
        # Prefer extracted traceback over full stderr
        error_msg = (
            state.execution_result.get("extracted_error")
            or state.execution_result.get("stderr")
            or error_msg
        )
    return remediate_spec_node(state, error_message=error_msg)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_aegis_graph():
    """Build and compile the Aegis LangGraph.

    Returns:
        A compiled ``StateGraph`` ready for ``invoke`` / ``stream`` calls.
    """
    builder = StateGraph(AegisState)

    # -- nodes ---------------------------------------------------------------
    builder.add_node("parse_intent", _wrap_parse_intent)
    builder.add_node("estimate_cost", estimate_cost_node)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("execute", execute_node)
    builder.add_node("run_evals", run_evals_node)
    builder.add_node("eval_gate", eval_gate_node)
    builder.add_node("remediate_spec", _wrap_remediate)
    builder.add_node("check_retries", check_retries_node)
    builder.add_node("write_report", write_report_node)

    # -- edges ---------------------------------------------------------------
    builder.set_entry_point("parse_intent")

    # Linear: parse_intent -> estimate_cost
    builder.add_edge("parse_intent", "estimate_cost")

    # estimate_cost --[budget_gate_routing]--> budget_gate (always)
    builder.add_conditional_edges("estimate_cost", budget_gate_routing)

    # budget_gate --[budget_decision_routing]--> generate_code | remediate_spec | END
    builder.add_conditional_edges("budget_gate", budget_decision_routing)

    # Linear: generate_code -> execute -> run_evals -> eval_gate
    builder.add_edge("generate_code", "execute")
    builder.add_edge("execute", "run_evals")
    builder.add_edge("run_evals", "eval_gate")

    # eval_gate --[eval_gate_routing]--> write_report | remediate_spec
    builder.add_conditional_edges("eval_gate", eval_gate_routing)

    # check_retries --[retry_routing]--> remediate_spec | write_report
    builder.add_conditional_edges("check_retries", retry_routing)

    # remediate_spec loops back to cost estimation
    builder.add_edge("remediate_spec", "estimate_cost")

    # write_report is the terminal node
    builder.add_edge("write_report", END)

    # -- compile -------------------------------------------------------------
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
