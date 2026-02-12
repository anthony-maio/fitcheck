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
from aegis.nodes.profiler import estimate_cost_node
from aegis.nodes.gate import budget_gate_node
from aegis.nodes.generator import generate_code_node
from aegis.nodes.evaluator import run_evals_node
from aegis.nodes.remediate import remediate_spec_node
from aegis.nodes.reporter import write_report_node

from aegis.edges.budget import budget_gate_routing, budget_decision_routing
from aegis.edges.retry import retry_routing
from aegis.edges.eval import eval_gate_routing


# ---------------------------------------------------------------------------
# Stub / adapter nodes
# ---------------------------------------------------------------------------

def execute_node(state: AegisState) -> AegisState:
    """Stub executor -- returns mock result for MVP."""
    if state.generated_code is None:
        event = RunEvent(
            phase="execute",
            status="failed",
            message="No code to execute",
        )
        return state.model_copy(update={"events": state.events + [event]})

    mock_result = {
        "metrics": {"train_loss": 1.2, "eval_loss": 1.5},
        "model_path": "/tmp/mock_model",
        "duration_sec": 300,
    }
    event = RunEvent(
        phase="execute",
        status="completed",
        message="Mock execution completed",
        data=mock_result,
    )
    return state.model_copy(
        update={"execution_result": mock_result, "events": state.events + [event]}
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
    """Wrap ``parse_intent_node`` for LangGraph (uses default input)."""
    return parse_intent_node(state, user_input="Fine-tune tinyllama with LoRA")


def _wrap_remediate(state: AegisState) -> AegisState:
    """Wrap ``remediate_spec_node`` for LangGraph.

    Extracts a meaningful error message from the most recent execution or
    evaluation result so that the remediation heuristic can act on it.
    """
    error_msg = "Generic training failure"
    if state.execution_result and state.execution_result.get("stderr"):
        error_msg = state.execution_result["stderr"]
    elif state.eval_result and state.eval_result.get("failures"):
        error_msg = "; ".join(state.eval_result["failures"])
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
