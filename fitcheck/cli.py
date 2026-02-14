"""fitcheck CLI -- know before you train.

Typer-based command-line interface. Entry point: `fitcheck plan`.
"""

from __future__ import annotations

from typing import Optional

import typer

from fitcheck.hardware.registry import get_hardware
from fitcheck.hub.resolver import resolve_model
from fitcheck.models.profiles import LoRAConfig, TrainingMethod
from fitcheck.models.results import PlanReport
from fitcheck.profilers.vram.components import get_trainable_params
from fitcheck.report.formatter import print_report
from fitcheck.solver import ConfigSolver

app = typer.Typer(
    name="fitcheck",
    help="Know before you train -- VRAM estimation for LLM fine-tuning.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """fitcheck -- know before you train."""


@app.command()
def plan(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID"),
    method: str = typer.Option(..., "--method", help="Training method: full, lora, qlora"),
    gpu: str = typer.Option(..., "--gpu", "-g", help="GPU name or alias (e.g. 3090, h100)"),
    seq_len: int = typer.Option(512, "--seq-len", help="Sequence length for estimation"),
    lora_rank: int = typer.Option(16, "--lora-rank", help="LoRA rank"),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override solver batch search with a fixed value"
    ),
    eval_seq_len: Optional[int] = typer.Option(
        None, "--eval-seq-len", help="Max eval sequence length for KV-cache spike"
    ),
) -> None:
    """Estimate VRAM usage and find optimal training config."""
    # Resolve training method
    try:
        training_method = TrainingMethod(method.lower())
    except ValueError:
        typer.echo(
            f"Error: Unknown method '{method}'. Choose from: full, lora, qlora",
            err=True,
        )
        raise typer.Exit(code=1)

    # Resolve hardware
    try:
        hardware = get_hardware(gpu)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Resolve model from HF Hub
    try:
        model_profile = resolve_model(model)
    except Exception as e:
        typer.echo(f"Error resolving model '{model}': {e}", err=True)
        raise typer.Exit(code=1)

    # Build LoRA config
    lora_config = LoRAConfig(rank=lora_rank)

    # Run solver
    solver = ConfigSolver()

    if batch_size is not None:
        solver_result = solver.estimate_fixed(
            model=model_profile,
            hardware=hardware,
            method=training_method,
            batch_size=batch_size,
            seq_len=seq_len,
            lora_config=lora_config,
            eval_seq_len=eval_seq_len,
        )
    else:
        solver_result = solver.solve(
            model=model_profile,
            hardware=hardware,
            method=training_method,
            seq_len=seq_len,
            lora_config=lora_config,
            eval_seq_len=eval_seq_len,
        )

    # Compute trainable params / pct
    trainable = get_trainable_params(model_profile, training_method, lora_config)
    trainable_pct = (trainable / model_profile.total_params) * 100

    # TODO: integrate sanity checker via --dataset flag (fitcheck.profilers.sanity)

    # Assemble report
    report = PlanReport(
        model_id=model_profile.model_id,
        architecture_summary=model_profile.architecture,
        total_params_b=model_profile.total_params_b,
        vocab_size=model_profile.vocab_size,
        num_layers=model_profile.num_layers,
        seq_len_used=seq_len,
        seq_len_reasoning=f"--seq-len {seq_len}",
        hardware_name=hardware.name,
        total_vram_gb=hardware.total_vram_gb,
        overhead_gb=hardware.overhead_gb,
        usable_vram_gb=hardware.usable_vram_gb,
        method=training_method.value,
        trainable_params=trainable,
        trainable_pct=trainable_pct,
        solver_result=solver_result,
    )

    print_report(report)


if __name__ == "__main__":
    app()
