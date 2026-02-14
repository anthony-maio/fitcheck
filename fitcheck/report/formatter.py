"""Rich terminal report renderer for fitcheck plan output.

Renders a PlanReport as structured, aligned terminal output using Rich.
Designed to be screenshot-friendly: clean typography, subtle color,
no gratuitous decoration.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fitcheck.models.results import PlanReport, TrainingConfig
from fitcheck.solver import RECOMMENDED_HEADROOM_PCT


def format_report(report: PlanReport, *, width: int | None = None) -> str:
    """Render a PlanReport to a string of Rich-formatted terminal output."""
    buf = StringIO()
    console = Console(file=buf, width=width or 90, force_terminal=True)
    _render_all(console, report)
    return buf.getvalue()


def print_report(report: PlanReport) -> None:
    """Render and print a PlanReport directly to the terminal."""
    console = Console(width=90)
    _render_all(console, report)


def _render_all(console: Console, report: PlanReport) -> None:
    """Render all report sections to a console."""
    _render_header(console, report)
    _render_model_summary(console, report)
    _render_dataset_summary(console, report)
    _render_hardware_summary(console, report)
    _render_training_summary(console, report)
    _render_vram_breakdown(console, report)
    _render_recommended_config(console, report)
    _render_aggressive_config(console, report)
    _render_risks(console, report)
    _render_fallbacks(console, report)


# --- Section Renderers ---


def _render_header(console: Console, report: PlanReport) -> None:
    title = Text("fitcheck plan", style="bold cyan")
    console.print()
    console.print(Panel(title, expand=False, border_style="dim"))
    console.print()


def _render_model_summary(console: Console, report: PlanReport) -> None:
    console.print(Text("Model", style="bold"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=14)
    table.add_column()
    table.add_row("Model ID", report.model_id)
    table.add_row("Architecture", report.architecture_summary)
    table.add_row("Parameters", f"{report.total_params_b:.2f}B")
    table.add_row("Vocab size", f"{report.vocab_size:,}")
    table.add_row("Layers", str(report.num_layers))
    console.print(table)
    console.print()


def _render_dataset_summary(console: Console, report: PlanReport) -> None:
    console.print(Text("Dataset", style="bold"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=14)
    table.add_column()

    if report.dataset_source and report.dataset_source != "none":
        table.add_row("Source", report.dataset_source)
        if report.dataset_rows:
            table.add_row("Rows", f"{report.dataset_rows:,}")
        table.add_row("Format", report.dataset_format)
        if report.seq_len_stats:
            stats = report.seq_len_stats
            table.add_row(
                "Seq lengths",
                f"p50={stats.get('p50', '?')} p95={stats.get('p95', '?')} "
                f"max={stats.get('max', '?')}",
            )
    else:
        table.add_row("Source", "No dataset provided")

    table.add_row("Seq len used", f"{report.seq_len_used}")
    table.add_row("Reasoning", report.seq_len_reasoning)
    console.print(table)
    console.print()


def _render_hardware_summary(console: Console, report: PlanReport) -> None:
    console.print(Text("Hardware", style="bold"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=14)
    table.add_column()
    table.add_row("GPU", report.hardware_name)
    table.add_row("Total VRAM", f"{report.total_vram_gb:.1f} GB")
    table.add_row("Overhead", f"{report.overhead_gb:.1f} GB")
    table.add_row("Usable VRAM", f"{report.usable_vram_gb:.1f} GB")
    console.print(table)
    console.print()


def _render_training_summary(console: Console, report: PlanReport) -> None:
    console.print(Text("What You're Training", style="bold"))
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=14)
    table.add_column()
    table.add_row("Method", report.method.upper())
    table.add_row("Trainable", f"{report.trainable_pct:.2f}%")
    if report.trainable_params:
        trainable_m = report.trainable_params / 1e6
        table.add_row("Trainable params", f"{trainable_m:.1f}M")
    if report.samples_per_epoch and report.samples_per_epoch > 0:
        table.add_row("Samples/epoch", f"{report.samples_per_epoch:,.0f}")
    console.print(table)
    console.print()


def _render_vram_breakdown(console: Console, report: PlanReport) -> None:
    rec = report.solver_result.recommended
    breakdown = rec.vram_breakdown
    if breakdown is None:
        return

    console.print(Text("VRAM Breakdown", style="bold"))

    table = Table(box=None, padding=(0, 2))
    table.add_column("Component", style="dim", min_width=18)
    table.add_column("VRAM", justify="right", min_width=10)
    table.add_column("Detail", style="dim")

    for comp in breakdown.components:
        table.add_row(comp.name, comp.display, comp.description)

    # Totals
    table.add_row("", "", "")
    table.add_row(
        Text("Steady state", style="bold"),
        Text(f"{breakdown.steady_state_gb:.2f} GB", style="bold"),
        "",
    )
    table.add_row(
        "Dynamic margin",
        f"{breakdown.dynamic_margin_bytes / (1024**3):.2f} GB",
        "fragmentation + transient peaks",
    )
    table.add_row(
        Text("Range", style="bold"),
        Text(
            f"{breakdown.range_low_gb:.2f}-{breakdown.range_high_gb:.2f} GB",
            style="bold",
        ),
        "",
    )

    # Headroom
    headroom_gb = report.usable_vram_gb - breakdown.range_high_gb
    headroom_pct = headroom_gb / report.usable_vram_gb * 100
    if headroom_pct >= RECOMMENDED_HEADROOM_PCT:
        style = "green"
    elif headroom_pct >= 5:
        style = "yellow"
    else:
        style = "red"

    table.add_row(
        "Headroom",
        Text(f"{headroom_gb:.1f} GB ({headroom_pct:.0f}%)", style=style),
        f"of {report.usable_vram_gb:.1f} GB usable",
    )

    if breakdown.eval_kv_spike is not None:
        table.add_row("", "", "")
        table.add_row(
            "Eval KV-cache spike",
            breakdown.eval_kv_spike.display,
            breakdown.eval_kv_spike.description,
        )

    console.print(table)
    console.print()


def _render_recommended_config(console: Console, report: PlanReport) -> None:
    rec = report.solver_result.recommended

    if rec.reasoning.get("verdict") == "does_not_fit":
        console.print(
            Panel(
                Text(
                    f"DOES NOT FIT: {rec.reasoning.get('detail', '')}",
                    style="bold red",
                ),
                title="Recommended Config",
                border_style="red",
            )
        )
        console.print()
        return

    console.print(Text("Recommended Config", style="bold green"))
    _render_config_table(console, rec)
    console.print()


def _render_aggressive_config(console: Console, report: PlanReport) -> None:
    agg = report.solver_result.aggressive
    if agg is None:
        return

    console.print(Text("Aggressive Config", style="bold yellow"))
    console.print(
        Text("  Tighter fit -- may OOM on longer sequences", style="dim yellow"),
    )
    _render_config_table(console, agg)
    console.print()


def _render_config_table(console: Console, config: TrainingConfig) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=14)
    table.add_column()
    table.add_row("Micro batch", str(config.micro_batch_size))
    table.add_row("Grad accum", str(config.gradient_accumulation_steps))
    table.add_row("Effective batch", str(config.effective_batch_size))
    table.add_row("Seq len", str(config.seq_len))
    table.add_row("Grad ckpt", "yes" if config.gradient_checkpointing else "no")
    table.add_row("Optimizer", config.optimizer)
    if config.lora_rank is not None:
        table.add_row("LoRA rank", str(config.lora_rank))

    if config.vram_breakdown:
        bd = config.vram_breakdown
        table.add_row(
            "VRAM",
            f"{bd.range_low_gb:.1f}-{bd.range_high_gb:.1f} GB",
        )

    console.print(table)


def _render_risks(console: Console, report: PlanReport) -> None:
    warnings = report.solver_result.warnings
    if not warnings:
        return

    console.print(Text("Risks", style="bold red"))
    for w in warnings:
        console.print(f"  - {w}")
    console.print()


def _render_fallbacks(console: Console, report: PlanReport) -> None:
    fallbacks = report.solver_result.fallbacks
    if not fallbacks:
        return

    console.print(Text("Fallback Chain", style="bold"))
    console.print(
        Text("  If you hit OOM, try these in order:", style="dim"),
    )
    console.print()

    for i, fb in enumerate(fallbacks, 1):
        parts = [f"bs={fb.micro_batch_size}"]
        if fb.gradient_checkpointing:
            parts.append("grad_ckpt=on")
        if fb.lora_rank is not None:
            parts.append(f"rank={fb.lora_rank}")
        if fb.vram_breakdown:
            parts.append(f"~{fb.vram_breakdown.steady_state_gb:.1f} GB")
        console.print(f"  {i}. {', '.join(parts)}")

    console.print()
