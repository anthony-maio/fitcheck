"""Streamlit UI for the Aegis-ML fine-tuning orchestrator.

Provides a chat-driven interface with a status sidebar, cost metrics,
phase tracker, event log, HITL budget approval, streaming progress,
and a report viewer with download support.
"""

import re
import time
import streamlit as st

from aegis.models.state import AegisState
from aegis.graph import build_aegis_graph

st.set_page_config(page_title="Aegis-ML", page_icon="\U0001f6e1\ufe0f", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling — Inter font, tighter spacing
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

.stMetric label {
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    opacity: 0.7;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 1.3rem;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    font-size: 0.9rem;
}

.stChatInput textarea {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# GPU cost reference ($/hr) for the comparison table
# ---------------------------------------------------------------------------

# ($/hr, relative throughput vs T4 for LLM training)
GPU_SPECS = {
    "T4":        {"rate": 0.59, "speed": 1.0},
    "L4":        {"rate": 0.80, "speed": 1.6},
    "A10G":      {"rate": 1.10, "speed": 2.2},
    "L40S":      {"rate": 1.95, "speed": 4.0},
    "A100-40GB": {"rate": 2.10, "speed": 5.5},
    "A100-80GB": {"rate": 2.50, "speed": 6.0},
    "H100":      {"rate": 3.95, "speed": 10.0},
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_aegis_graph()
if "aegis_state" not in st.session_state:
    st.session_state.aegis_state = {}
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-thread"
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        st.metric("Est. Duration", f"{ce.estimated_duration_min:.1f} min")
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
        "write_report",
    ]
    events = state.get("events", [])
    for phase in phases:
        if any(e.phase == phase and e.status == "completed" for e in events):
            st.success(phase)
        elif any(e.phase == phase and e.status == "failed" for e in events):
            st.error(f"\u274c {phase}")
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

    # -- GPU cost comparison ------------------------------------------------
    st.markdown("### GPU Cost Comparison")
    if state.get("cost_estimate"):
        ce = state["cost_estimate"]
        # Base duration at the selected GPU's speed
        selected_gpu = state.get("spec", {})
        selected_gpu_name = getattr(selected_gpu, "target_gpu", "a10g").upper()
        base_dur_min = ce.estimated_duration_min
        # Find the speed of the selected GPU to normalize
        selected_speed = 1.0
        for gn, gs in GPU_SPECS.items():
            if gn.lower().replace("-", "") == selected_gpu_name.lower().replace("-", ""):
                selected_speed = gs["speed"]
                break

        best_cost = float("inf")
        rows: list[dict] = []
        for gpu_name, gs in GPU_SPECS.items():
            # Duration scales inversely with relative speed
            adj_dur_min = base_dur_min * (selected_speed / gs["speed"])
            adj_dur_hr = adj_dur_min / 60
            total_cost = gs["rate"] * adj_dur_hr
            rows.append({
                "name": gpu_name,
                "rate": gs["rate"],
                "speed": gs["speed"],
                "dur_min": adj_dur_min,
                "total": total_cost,
            })
            if total_cost < best_cost:
                best_cost = total_cost

        for r in rows:
            tag = ""
            if r["total"] <= best_cost * 1.01:
                tag = " **Best value**"
            elif r["total"] > best_cost * 2.5:
                tag = " *Overpriced*"
            st.caption(
                f"{'**' if tag == ' **Best value**' else ''}"
                f"{r['name']}: ${r['total']:.4f} "
                f"({r['dur_min']:.0f}min, ${r['rate']}/hr)"
                f"{'**' if tag == ' **Best value**' else ''}"
                f"{tag}"
            )
    else:
        for gpu_name, gs in GPU_SPECS.items():
            st.caption(f"{gpu_name}: ${gs['rate']}/hr ({gs['speed']}x speed)")

    st.markdown("---")

    # -- Reset button -------------------------------------------------------
    if st.button("\U0001f504 Reset"):
        st.session_state.aegis_state = {}
        st.session_state.awaiting_approval = False
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Fine-Tuning Orchestrator")
st.markdown("Describe your fine-tuning job and Aegis will handle the rest safely.")

# -- Render persisted chat history ------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ---------------------------------------------------------------------------
# HITL budget approval dialog (D2)
# ---------------------------------------------------------------------------

def _check_budget_interrupted(result: dict) -> bool:
    """Return True if the budget gate is awaiting human approval."""
    for event in result.get("events", []):
        if event.phase == "budget_gate" and event.status == "interrupted":
            return True
    return False


if st.session_state.awaiting_approval:
    st.warning("\u23f8\ufe0f Budget gate requires human approval")
    ce = state.get("cost_estimate")
    if ce:
        st.info(
            f"Estimated cost: **${ce.estimated_cost_usd:.4f}** "
            f"(soft threshold: ${state['budget_policy'].soft_threshold_usd:.2f}, "
            f"hard limit: ${state['budget_policy'].max_budget_usd:.2f})"
        )

    col1, col2, col3 = st.columns(3)
    decision = None
    with col1:
        if st.button("\u2705 Approve", use_container_width=True):
            decision = "approve"
    with col2:
        if st.button("\U0001f527 Optimize", use_container_width=True):
            decision = "optimize"
    with col3:
        if st.button("\u274c Cancel", use_container_width=True):
            decision = "cancel"

    if decision:
        graph = st.session_state.graph
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        graph.update_state(config, {"human_decision": decision})
        result = graph.invoke(None, config=config)
        st.session_state.aegis_state = result
        st.session_state.awaiting_approval = False
        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline phase metadata for the progress panel
# ---------------------------------------------------------------------------

PHASE_INFO = {
    "parse_intent": {
        "label": "Parse Intent",
        "desc": "Extracting training spec from natural language via LLM",
        "icon": "\U0001f9e0",
    },
    "estimate_cost": {
        "label": "Estimate Cost",
        "desc": "Profiling GPU cost, VRAM, and duration",
        "icon": "\U0001f4b0",
    },
    "budget_gate": {
        "label": "Budget Gate",
        "desc": "Evaluating cost against FinOps policy",
        "icon": "\U0001f6e1\ufe0f",
    },
    "generate_code": {
        "label": "Generate Code",
        "desc": "Rendering HuggingFace training script from spec",
        "icon": "\U0001f40d",
    },
    "execute": {
        "label": "Execute Training",
        "desc": "Running training on Modal GPU infrastructure",
        "icon": "\U0001f680",
    },
    "run_evals": {
        "label": "Run Evaluations",
        "desc": "Checking loss convergence, overfitting, safety canary",
        "icon": "\U0001f50d",
    },
    "eval_gate": {
        "label": "Eval Gate",
        "desc": "Routing based on evaluation results",
        "icon": "\u2696\ufe0f",
    },
    "check_retries": {
        "label": "Check Retries",
        "desc": "Evaluating retry budget for auto-healing",
        "icon": "\U0001f504",
    },
    "remediate_spec": {
        "label": "Remediate Spec",
        "desc": "Auto-healing: adjusting training parameters",
        "icon": "\U0001fa79",
    },
    "write_report": {
        "label": "Write Report",
        "desc": "Generating audit trail and final report",
        "icon": "\U0001f4c4",
    },
}

PHASE_ORDER = [
    "parse_intent", "estimate_cost", "budget_gate", "generate_code",
    "execute", "run_evals", "eval_gate", "write_report",
]

# ---------------------------------------------------------------------------
# Chat input & streaming execution (D1 + D3)
# ---------------------------------------------------------------------------

user_input = st.chat_input("e.g., 'Fine-tune tinyllama with LoRA for 3 epochs'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = AegisState(user_input=user_input)

    # -- Live progress panel ------------------------------------------------
    progress_container = st.container()
    with progress_container:
        st.markdown("### \u2699\ufe0f Pipeline Running")
        progress_bar = st.progress(0)
        status_area = st.empty()
        log_area = st.empty()

    log_lines: list[str] = []
    retry_num = 0
    start_time = time.time()
    final_result = None
    _event_cursor = [0]  # mutable container to avoid nonlocal

    def _elapsed() -> str:
        return f"{time.time() - start_time:.1f}s"

    def _render_status(current_node: str | None = None):
        """Update the current-phase banner."""
        if current_node and current_node in PHASE_INFO:
            info = PHASE_INFO[current_node]
            status_area.markdown(
                f"**{info['icon']} {info['label']}** \u2014 {info['desc']}  \n"
                f"`{_elapsed()} elapsed`"
            )
        elif current_node:
            status_area.markdown(
                f"**\u2699\ufe0f {current_node}**  \n`{_elapsed()} elapsed`"
            )

    def _render_log():
        """Render scrolling log with fixed height."""
        log_text = "\n".join(log_lines[-50:])  # keep last 50 lines
        log_area.markdown(
            f'<div style="height:300px;overflow-y:auto;background:#0e1117;'
            f'border:1px solid #333;border-radius:6px;padding:12px;'
            f'font-family:monospace;font-size:0.82rem;line-height:1.5;'
            f'white-space:pre-wrap;">{log_text}</div>',
            unsafe_allow_html=True,
        )

    def _extract_new_events(node_output: dict) -> list:
        """Get events added by this node."""
        events = node_output.get("events", [])
        new = events[_event_cursor[0]:]
        _event_cursor[0] = len(events)
        return new

    # Show immediate feedback
    log_lines.append(f'<span style="color:#888">[{_elapsed()}]</span> Starting pipeline...')
    _render_status("parse_intent")
    _render_log()

    prev_time = start_time
    node_count = 0
    for chunk in st.session_state.graph.stream(initial_state, config=config):
        for node_name, node_output in chunk.items():
            now = time.time()
            dur = now - prev_time
            prev_time = now
            node_count += 1
            final_result = node_output

            info = PHASE_INFO.get(node_name, {"label": node_name, "icon": "\u2699\ufe0f"})

            # Detect retry loop
            if node_name == "remediate_spec":
                retry_num += 1

            # Extract event messages from this node's output
            new_events = _extract_new_events(node_output)
            for ev in new_events:
                status_icon = {
                    "completed": '<span style="color:#4ade80">\u2705</span>',
                    "failed": '<span style="color:#f87171">\u274c</span>',
                    "interrupted": '<span style="color:#fbbf24">\u23f8\ufe0f</span>',
                }.get(ev.status, '<span style="color:#888">\u2022</span>')

                ts = f'<span style="color:#888">[{_elapsed()}]</span>'

                if ev.status == "failed":
                    log_lines.append(
                        f'{ts} {status_icon} <span style="color:#f87171">'
                        f'<b>{ev.phase}</b>: {ev.message}</span>'
                    )
                elif ev.phase == "remediate_spec":
                    log_lines.append(
                        f'{ts} {status_icon} <span style="color:#fbbf24">'
                        f'<b>Retry {retry_num}/{node_output.get("max_retries", 3)}</b>'
                        f': {ev.message}</span>'
                    )
                else:
                    log_lines.append(
                        f'{ts} {status_icon} <b>{ev.phase}</b>: {ev.message}'
                    )

                # Show key data inline
                if ev.data:
                    if ev.phase == "parse_intent" and "parsed_spec" in ev.data:
                        spec_d = ev.data["parsed_spec"]
                        log_lines.append(
                            f'<span style="color:#888">     model={spec_d.get("model_name")} '
                            f'method={spec_d.get("method")} lr={spec_d.get("learning_rate")} '
                            f'gpu={spec_d.get("target_gpu")}</span>'
                        )
                    if ev.phase == "estimate_cost" and "estimated_seconds" in ev.data:
                        log_lines.append(
                            f'<span style="color:#888">     params={ev.data.get("model_params_b", "?")}B '
                            f'moe={ev.data.get("is_moe", False)} '
                            f'vram={ev.data.get("base_model_vram_gb", "?")}+'
                            f'{ev.data.get("optimizer_vram_gb", "?")}+'
                            f'{ev.data.get("activation_vram_gb", "?")} GB</span>'
                        )
                    if ev.phase == "execute" and ev.status == "failed":
                        # Prefer extracted_error (the actual traceback)
                        err_text = (
                            ev.data.get("extracted_error")
                            or ev.data.get("stderr")
                            or ""
                        )
                        # Show last 600 chars of the actual error
                        if err_text:
                            err_display = err_text[-600:].replace("<", "&lt;").replace(">", "&gt;")
                            log_lines.append(
                                f'<span style="color:#f87171;font-size:0.75rem">'
                                f'     {err_display}</span>'
                            )
                    if ev.phase == "remediate_spec" and ev.data and ev.data.get("is_environment_error"):
                        log_lines.append(
                            f'<span style="color:#fbbf24;font-size:0.75rem">'
                            f'     This is an environment issue, not a training parameter problem.</span>'
                        )

            # Progress bar: estimate based on happy path length, cap at 95% during loop
            happy_path_len = len(PHASE_ORDER)
            frac = min(node_count / (happy_path_len + retry_num * 7), 0.95)
            progress_bar.progress(frac)

            # Show next phase
            _render_status(None)  # clear while we figure out next
            # Peek at what comes next based on node ordering
            done_names = {node_name}
            next_phase = None
            for ph in PHASE_ORDER:
                if ph not in done_names:
                    next_phase = ph
                    break
            _render_status(next_phase or node_name)
            _render_log()

    # Done
    elapsed = time.time() - start_time
    progress_bar.progress(1.0)
    if retry_num > 0:
        status_area.markdown(
            f"\U0001f3c1 **Pipeline complete** \u2014 `{elapsed:.1f}s` "
            f"({retry_num} retry{'s' if retry_num != 1 else ''})"
        )
    else:
        status_area.markdown(f"\u2705 **Pipeline complete** \u2014 `{elapsed:.1f}s total`")

    if final_result is not None:
        st.session_state.aegis_state = final_result
        if _check_budget_interrupted(final_result):
            st.session_state.awaiting_approval = True

        # Build assistant summary message
        spec = final_result.get("spec")
        ce = final_result.get("cost_estimate")
        passed = final_result.get("eval_result", {}).get("passed")
        parts = []
        if spec:
            parts.append(f"**{spec.method}** on `{spec.model_name}`")
        if ce:
            parts.append(f"est. ${ce.estimated_cost_usd:.4f}, {ce.estimated_vram_gb:.1f}GB VRAM")
        if passed is True:
            parts.append("evals passed")
        elif passed is False:
            parts.append("evals failed")
        summary = " | ".join(parts) if parts else "Pipeline finished."
        st.session_state.chat_history.append({"role": "assistant", "content": summary})

    time.sleep(1.5)
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
        if "metrics" in result and result["metrics"]:
            for key, value in result["metrics"].items():
                if isinstance(value, (int, float)):
                    st.metric(key, f"{value:.4f}")
    with col2:
        duration = result.get("duration_sec")
        if duration is not None:
            st.metric("Duration", f"{duration / 60:.1f} min")

    # D5: Loss curve chart — parse step-by-step loss from stdout
    stdout = result.get("stdout", "")
    loss_steps = re.findall(r"'loss':\s*([\d.]+)", stdout)
    if loss_steps:
        loss_values = [float(v) for v in loss_steps]
        st.markdown("#### Training Loss Curve")
        st.line_chart(loss_values)

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

# -- D4: Failed events with details ----------------------------------------
failed_events = [e for e in events if e.status == "failed"]
if failed_events:
    st.markdown("---")
    st.subheader("\u26a0\ufe0f Errors & Failures")
    for event in failed_events:
        with st.expander(f"\u274c {event.phase}: {event.message}", expanded=False):
            st.text(f"Timestamp: {event.timestamp}")
            if event.data:
                st.json(event.data)

# -- Final Report -----------------------------------------------------------
if state.get("final_report"):
    st.markdown("---")
    st.subheader("\U0001f4c4 Final Report")
    st.markdown(state["final_report"])
    report_data = state["final_report"]
    st.download_button(
        label="\U0001f4e5 Download Report",
        data=report_data,
        file_name="aegis_report.md",
        mime="text/markdown",
        key="download_report",
    )
