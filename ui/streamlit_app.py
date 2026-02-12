"""Streamlit UI for the Aegis-ML fine-tuning orchestrator.

Provides a chat-driven interface with a status sidebar, cost metrics,
phase tracker, event log, and a report viewer with download support.
"""

import streamlit as st

from aegis.models.state import AegisState
from aegis.graph import build_aegis_graph

st.set_page_config(page_title="Aegis-ML", page_icon="\U0001f6e1\ufe0f", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_aegis_graph()
if "aegis_state" not in st.session_state:
    st.session_state.aegis_state = {}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-thread"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("\U0001f6e1\ufe0f Aegis-ML")
    st.markdown("### Production-Grade AI Reliability Engineer")
    st.markdown("---")

    state = st.session_state.aegis_state

    # -- Cost metrics -------------------------------------------------------
    if state.get("cost_estimate"):
        ce = state["cost_estimate"]
        st.metric("Est. Cost", f"${ce.estimated_cost_usd:.4f}")
        st.metric("VRAM", f"{ce.estimated_vram_gb:.1f} GB")
    else:
        st.caption("No cost estimate yet")

    st.markdown("---")

    # -- Phase tracker ------------------------------------------------------
    st.markdown("### Pipeline Phase")
    phases = [
        "parse_intent",
        "estimate_cost",
        "budget_gate",
        "generate_code",
        "execute",
        "run_evals",
    ]
    events = state.get("events", [])
    for phase in phases:
        if any(e.phase == phase and e.status == "completed" for e in events):
            st.success(phase)
        elif any(e.phase == phase for e in events):
            st.info(f"\u27a4 {phase}")
        else:
            st.caption(phase)

    st.markdown("---")

    # -- Event log ----------------------------------------------------------
    st.markdown("### Event Log")
    for event in reversed(events[-10:]):
        icon = {
            "completed": "\u2705",
            "failed": "\u274c",
            "started": "\u25b6\ufe0f",
            "interrupted": "\u23f8\ufe0f",
        }.get(event.status, "\u2022")
        st.caption(f"{icon} {event.phase}: {event.message}")

    st.markdown("---")

    # -- Reset button -------------------------------------------------------
    if st.button("\U0001f504 Reset"):
        st.session_state.aegis_state = {}
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Fine-Tuning Orchestrator")
st.markdown("Describe your fine-tuning job and Aegis will handle the rest safely.")

user_input = st.chat_input("e.g., 'Fine-tune tinyllama with LoRA for 3 epochs'")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = st.session_state.graph.invoke(AegisState(), config=config)
    st.session_state.aegis_state = result
    st.rerun()

# ---------------------------------------------------------------------------
# Display sections (populated after a graph run)
# ---------------------------------------------------------------------------

# -- Training Specification -------------------------------------------------
if state.get("spec"):
    st.markdown("---")
    st.subheader("\U0001f4cb Training Specification")
    spec = state["spec"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Method", spec.method)
        st.metric("Model", spec.model_name.split("/")[-1])
    with col2:
        st.metric("Batch Size", spec.micro_batch_size)
        st.metric("Epochs", spec.num_epochs)
    with col3:
        st.metric("Learning Rate", f"{spec.learning_rate:.0e}")
        st.metric("Seq Len", spec.seq_len)

# -- Generated Code ---------------------------------------------------------
if state.get("generated_code"):
    st.markdown("---")
    st.subheader("\U0001f40d Generated Training Script")
    with st.expander("View code", expanded=False):
        st.code(state["generated_code"], language="python")

# -- Execution Results ------------------------------------------------------
if state.get("execution_result"):
    st.markdown("---")
    st.subheader("\U0001f4ca Execution Results")
    result = state["execution_result"]
    col1, col2 = st.columns(2)
    with col1:
        if "metrics" in result:
            for key, value in result["metrics"].items():
                st.metric(key, f"{value:.4f}")
    with col2:
        if "duration_sec" in result:
            st.metric("Duration", f"{result['duration_sec'] / 60:.1f} min")

# -- Evaluation Results -----------------------------------------------------
if state.get("eval_result"):
    st.markdown("---")
    st.subheader("\U0001f50d Evaluation Results")
    eval_result = state["eval_result"]
    if eval_result.get("passed"):
        st.success("\u2705 All evaluations passed!")
    else:
        st.error("\u274c Some evaluations failed")
        for failure in eval_result.get("failures", []):
            st.warning(f"- {failure}")

# -- Final Report -----------------------------------------------------------
if state.get("final_report"):
    st.markdown("---")
    st.subheader("\U0001f4c4 Final Report")
    st.markdown(state["final_report"])
    st.download_button(
        label="\U0001f4e5 Download Report",
        data=state["final_report"],
        file_name="aegis_report.md",
        mime="text/markdown",
    )
