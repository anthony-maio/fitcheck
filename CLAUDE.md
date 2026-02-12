# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aegis-ML is a LangGraph-based "Production-Grade AI Reliability Engineer" that orchestrates LLM fine-tuning workflows with FinOps cost gates, sandboxed execution, bounded auto-healing, and evaluation safety checks. Target audience: hiring managers evaluating production ML engineering skills.

## Key Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/nodes/test_intent.py -v

# Run specific test
pytest tests/nodes/test_intent.py::test_parse_intent_creates_spec -v

# Run Streamlit UI
streamlit run ui/streamlit_app.py

# Format code
ruff format aegis tests
```

## Architecture

**Spec-first design**: `TrainingSpec` Pydantic model is the source of truth. All mutations go through immutable `model_copy()` updates — never mutate state in place.

**LangGraph state machine** with this flow:
```
parse_intent → estimate_cost → budget_gate → generate_code → execute → run_evals → eval_gate → write_report
                    ↑                                                                    |
                    └──────────────── remediate_spec ←───────────────────────────────────┘
```

**Bounded auto-healing**: All remediation loops respect `retry_count < max_retries` (default 3). Exhausted retries route to `write_report` (END).

**HITL budget gate**: Costs below `soft_threshold_usd` auto-approve. Costs between soft and hard threshold trigger human approval. Costs above `max_budget_usd` auto-cancel.

**Hybrid code generation**: Template-based (Jinja2) for MVP reliability. `CodeGenerator` ABC provides extensible hook for future LLM-based generation.

## Key Patterns

### Writing a Node

All nodes accept `AegisState` and return `AegisState` via `model_copy()`:

```python
def my_node(state: AegisState) -> AegisState:
    if state.spec is None:
        event = RunEvent(phase="my_node", status="failed", message="...")
        return state.model_copy(update={"events": state.events + [event]})

    result = do_work()
    event = RunEvent(phase="my_node", status="completed", message="...")
    return state.model_copy(update={"result_field": result, "events": state.events + [event]})
```

### Writing an Edge

Edges are routing functions returning node names as strings:

```python
def my_routing(state: AegisState) -> str:
    if condition:
        return "next_node"
    return "__end__"
```

### Event Logging

Every node MUST append a `RunEvent` to `state.events`. This produces the REPORT.md audit trail.

## Module Layout

- `aegis/models/state.py` — All Pydantic models (TrainingSpec, AegisState, RunEvent, etc.)
- `aegis/nodes/` — LangGraph node functions (one per file)
- `aegis/edges/` — Conditional routing functions
- `aegis/profilers/cost.py` — Deterministic cost/VRAM estimator
- `aegis/generators/` — Template-based code generation with ABC interface
- `aegis/executors/` — Modal/Docker execution runners
- `aegis/graph.py` — Main LangGraph construction and compilation
- `ui/streamlit_app.py` — Streamlit chat UI with status sidebar
- `templates/hf_training.py.j2` — Jinja2 HuggingFace training script template

## Testing

- All nodes: `tests/nodes/`
- All edges: `tests/edges/`
- Graph integration: `tests/test_graph.py`
- End-to-end: `tests/test_e2e.py`
- Tests use TDD pattern; always write tests before implementation
