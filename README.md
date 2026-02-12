# Aegis-ML

> **Production-Grade AI Reliability Engineer** — Safe, observable fine-tuning orchestration with FinOps gates and auto-healing.

Aegis-ML is a LangGraph-based orchestration layer that brings production discipline to LLM fine-tuning. It checks costs before running, sandboxes execution, validates outputs, and generates audit reports.

## Why Aegis-ML?

Fine-tuning LLMs is easy. Doing it **safely, cheaply, and observably** is hard:

- **Surprise cloud bills** — Cost estimation and approval gates before GPU time
- **Failed training runs** — Auto-healing with spec mutation and bounded retries
- **Unsafe execution** — Sandboxed training with Modal containers
- **No audit trail** — Complete lineage reports for compliance and review

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# Run the UI
streamlit run ui/streamlit_app.py
```

## Architecture

```
parse_intent → estimate_cost → budget_gate → generate_code → execute → run_evals → eval_gate → write_report
                    ↑                                                                    |
                    └──────────────── remediate_spec ←───────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| `TrainingSpec` | Pydantic model defining training intent |
| `CostProfiler` | Deterministic cost/VRAM estimation |
| `BudgetGate` | HITL approval for expensive runs |
| `TemplateCodeGenerator` | Jinja2-based script generation |
| `ModalExecutor` | Remote GPU execution (sandboxed) |
| `EvalNode` | Post-training validation (loss, safety) |
| `ReportGenerator` | Markdown lineage artifacts |

## Example Usage

```python
from aegis.graph import build_aegis_graph
from aegis.models.state import AegisState

graph = build_aegis_graph()
config = {"configurable": {"thread_id": "my-run"}}
result = graph.invoke(AegisState(), config)

print(result["final_report"])
```

## Development

```bash
pytest                                    # Run all tests
pytest tests/nodes/test_intent.py -v      # Run specific test file
ruff format aegis tests                   # Format code
```

## License

MIT
