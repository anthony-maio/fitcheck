# Aegis-ML Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Aegis-ML, a LangGraph-based "Production-Grade AI Reliability Engineer" that orchestrates fine-tuning workflows with FinOps gates, sandboxed execution, and evaluation safety checks.

**Architecture:** Spec-first design where `TrainingSpec` Pydantic model is the source of truth. LangGraph state machine with bounded auto-healing loops, HITL budget gates, and template-based code generation (extensible to LLM). Execution via Modal (remote) or local Docker.

**Tech Stack:** LangGraph, Pydantic, Modal, Streamlit, Jinja2, LangSmith

---

## Phase 1: Foundation & Data Models

### Task 1.1: Core Pydantic State Models

**Files:**
- Create: `aegis/__init__.py`
- Create: `aegis/models/__init__.py`
- Create: `aegis/models/state.py`
- Create: `tests/models/test_state.py`

**Step 1: Write the failing test**

```python
# tests/models/test_state.py
import pytest
from aegis.models.state import (
    TrainingSpec, BudgetPolicy, CostEstimate, RunEvent, AegisState
)
from datetime import datetime

def test_training_spec_creation():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        num_epochs=3,
        micro_batch_size=4,
        target_gpu="a10g"
    )
    assert spec.method == "lora"
    assert spec.model_name == "tinyllama/tinyllama-272m"
    assert spec.micro_batch_size == 4

def test_budget_policy_defaults():
    policy = BudgetPolicy()
    assert policy.max_budget_usd == 5.00
    assert policy.soft_threshold_usd == 2.00
    assert policy.allow_auto_approve is True

def test_cost_estimate_fields():
    estimate = CostEstimate(
        estimated_cost_usd=0.0150,
        estimated_vram_gb=8.5,
        estimated_duration_min=5.2,
        cost_breakdown={"gpu": "a10g", "steps": 1000}
    )
    assert estimate.estimated_cost_usd == 0.0150
    assert estimate.estimated_vram_gb == 8.5

def test_run_event_creation():
    event = RunEvent(
        phase="estimate",
        status="completed",
        message="Cost estimated successfully"
    )
    assert event.phase == "estimate"
    assert isinstance(event.timestamp, datetime)

def test_aegis_state_initialization():
    state = AegisState()
    assert state.spec is None
    assert state.events == []
    assert state.retry_count == 0
    assert state.max_retries == 3

def test_aegis_state_immutability_pattern():
    """Verify that state updates use model_copy for immutability"""
    state = AegisState(retry_count=0)
    new_state = state.model_copy(update={"retry_count": 1})
    assert state.retry_count == 0  # Original unchanged
    assert new_state.retry_count == 1  # New state updated
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_state.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis'"

**Step 3: Write minimal implementation**

```python
# aegis/__init__.py
__version__ = "0.1.0"

# aegis/models/__init__.py
from aegis.models.state import (
    TrainingSpec,
    BudgetPolicy,
    CostEstimate,
    RunEvent,
    AegisState,
)

__all__ = [
    "TrainingSpec",
    "BudgetPolicy",
    "CostEstimate",
    "RunEvent",
    "AegisState",
]

# aegis/models/state.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from datetime import datetime


class TrainingSpec(BaseModel):
    """User's training intent - extracted by LLM, validated by profiler."""
    method: Literal["full_finetune", "lora", "qlora"]
    model_name: str
    dataset_path: str
    num_epochs: int = 3
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    seq_len: int = 512
    target_gpu: str = "a10g"


class BudgetPolicy(BaseModel):
    """FinOps gate configuration."""
    max_budget_usd: float = 5.00
    soft_threshold_usd: float = 2.00
    allow_auto_approve: bool = True


class CostEstimate(BaseModel):
    """Output of cost profiler."""
    estimated_cost_usd: float
    estimated_vram_gb: float
    estimated_duration_min: float
    cost_breakdown: dict[str, Any]


class RunEvent(BaseModel):
    """Append-only log for auditability."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    phase: str
    status: Literal["started", "completed", "failed", "interrupted"]
    message: str
    data: Optional[dict[str, Any]] = None


class AegisState(BaseModel):
    """Complete graph state - immutable updates via .model_copy()."""
    spec: Optional[TrainingSpec] = None
    budget_policy: BudgetPolicy = Field(default_factory=BudgetPolicy)
    cost_estimate: Optional[CostEstimate] = None
    events: list[RunEvent] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    generated_code: Optional[str] = None
    execution_result: Optional[dict[str, Any]] = None
    eval_result: Optional[dict[str, Any]] = None
    human_decision: Optional[Literal["approve", "optimize", "cancel"]] = None
    final_report: Optional[str] = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_state.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add aegis/ tests/models/
git commit -m "feat: add core Pydantic state models"
```

---

### Task 1.2: Cost Profiler

**Files:**
- Create: `aegis/profilers/__init__.py`
- Create: `aegis/profilers/cost.py`
- Create: `tests/profilers/test_cost.py`

**Step 1: Write the failing test**

```python
# tests/profilers/test_cost.py
import pytest
from aegis.models.state import TrainingSpec, CostEstimate
from aegis.profilers.cost import estimate_training_cost, GPU_PRICING


def test_gpu_pricing_constants():
    assert "a10g" in GPU_PRICING
    assert "t4" in GPU_PRICING
    assert GPU_PRICING["a10g"] == 0.000306  # $1.10/hr


def test_estimate_cost_basic_lora():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        num_epochs=3,
        micro_batch_size=4,
        target_gpu="a10g"
    )
    estimate = estimate_training_cost(spec)

    assert isinstance(estimate, CostEstimate)
    assert estimate.estimated_cost_usd > 0
    assert estimate.estimated_cost_usd < 1.0  # Should be cheap
    assert estimate.estimated_vram_gb > 0
    assert estimate.estimated_duration_min > 0


def test_estimate_cost_breakdown_contains_required_fields():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        target_gpu="a10g"
    )
    estimate = estimate_training_cost(spec)

    assert "gpu" in estimate.cost_breakdown
    assert "gpu_price_usd_per_sec" in estimate.cost_breakdown
    assert "estimated_seconds" in estimate.cost_breakdown
    assert "effective_batch_size" in estimate.cost_breakdown
    assert "total_steps" in estimate.cost_breakdown


def test_estimate_cost_scales_with_batch_size():
    spec_small = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=2,
        target_gpu="a10g"
    )
    spec_large = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=8,
        target_gpu="a10g"
    )

    estimate_small = estimate_training_cost(spec_small)
    estimate_large = estimate_training_cost(spec_large)

    # Smaller batch should cost similar (same samples) but slightly more overhead
    # Duration should be very similar (same number of samples processed)
    assert abs(estimate_small.estimated_duration_min - estimate_large.estimated_duration_min) < 1


def test_estimate_cost_different_gpu():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        target_gpu="a10g"
    )

    estimate_a10g = estimate_training_cost(spec)

    spec_t4 = spec.model_copy(update={"target_gpu": "t4"})
    estimate_t4 = estimate_training_cost(spec_t4)

    # T4 is cheaper per second, so total cost should be lower
    assert estimate_t4.estimated_cost_usd < estimate_a10g.estimated_cost_usd


def test_vram_estimation_includes_components():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=4,
        seq_len=512,
        target_gpu="a10g"
    )
    estimate = estimate_training_cost(spec)

    # VRAM should account for model weights, optimizer, and activations
    # For 272M model, should be several GB minimum
    assert estimate.estimated_vram_gb > 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/profilers/test_cost.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.profilers'"

**Step 3: Write minimal implementation**

```python
# aegis/profilers/__init__.py
from aegis.profilers.cost import estimate_training_cost, GPU_PRICING

__all__ = ["estimate_training_cost", "GPU_PRICING"]

# aegis/profilers/cost.py
from aegis.models.state import TrainingSpec, CostEstimate

# GPU pricing (USD/second) - source: modal.com/pricing
GPU_PRICING = {
    "a10g": 0.000306,  # $1.10/hr
    "t4": 0.000126,    # $0.45/hr
    "a100": 0.000793,  # $2.85/hr
}


def estimate_training_cost(spec: TrainingSpec) -> CostEstimate:
    """
    Deterministic cost estimation based on spec parameters.

    Formula: (samples / batch_size * grad_accum * num_epochs) / throughput * gpu_price
    Default throughput: 1000 samples/sec for 272M models (conservative)
    """
    # Assume 10k samples (small dataset for demo)
    num_samples = 10000

    # Estimate training steps
    effective_batch_size = spec.micro_batch_size * spec.gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * spec.num_epochs

    # Estimate duration (conservative: 1000 samples/sec)
    samples_per_sec = 1000
    estimated_seconds = (num_samples * spec.num_epochs) / samples_per_sec

    # Add 20% overhead for data loading, checkpointing
    estimated_seconds *= 1.2

    # Calculate cost
    gpu_price = GPU_PRICING.get(spec.target_gpu, GPU_PRICING["a10g"])
    estimated_cost = estimated_seconds * gpu_price

    # VRAM estimation (simplified)
    base_model_vram = 1.0  # ~1GB for 272M model
    optimizer_vram = base_model_vram * 4  # AdamW states
    activation_vram = (spec.micro_batch_size * spec.seq_len * 32) / 1e9  # fp32 bytes
    total_vram = base_model_vram + optimizer_vram + activation_vram

    return CostEstimate(
        estimated_cost_usd=round(estimated_cost, 4),
        estimated_vram_gb=round(total_vram, 2),
        estimated_duration_min=round(estimated_seconds / 60, 1),
        cost_breakdown={
            "gpu": spec.target_gpu,
            "gpu_price_usd_per_sec": gpu_price,
            "estimated_seconds": round(estimated_seconds, 1),
            "effective_batch_size": effective_batch_size,
            "total_steps": total_steps,
        }
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/profilers/test_cost.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add aegis/profilers/ tests/profilers/
git commit -m "feat: add deterministic cost profiler with GPU pricing"
```

---

## Phase 2: Code Generation (Template-Based)

### Task 2.1: Training Script Template

**Files:**
- Create: `templates/hf_training.py.j2`
- Create: `tests/generators/test_templates.py`

**Step 1: Write the failing test**

```python
# tests/generators/test_templates.py
import pytest
from jinja2 import Template
from aegis.models.state import TrainingSpec
from aegis.generators.template import TemplateCodeGenerator


def test_template_generator_exists():
    generator = TemplateCodeGenerator()
    assert generator is not None


def test_generate_basic_training_script():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        num_epochs=3,
        micro_batch_size=4,
        learning_rate=5e-5,
        target_gpu="a10g"
    )

    generator = TemplateCodeGenerator()
    code = generator.generate(spec)

    assert "from transformers import" in code
    assert "TrainingArguments" in code
    assert "LoraConfig" in code
    assert spec.model_name in code
    assert str(spec.learning_rate) in code


def test_generated_script_has_main_entrypoint():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl"
    )

    generator = TemplateCodeGenerator()
    code = generator.generate(spec)

    assert "if __name__" in code
    assert "main()" in code


def test_full_finetune_generates_different_code():
    spec = TrainingSpec(
        method="full_finetune",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl"
    )

    generator = TemplateCodeGenerator()
    code = generator.generate(spec)

    # Should NOT have LoRA config
    assert "LoraConfig" not in code
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/generators/test_templates.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.generators'"

**Step 3: Write minimal implementation**

```jinja2
{# templates/hf_training.py.j2 #}
#!/usr/bin/env python3
"""
Auto-generated training script for {{ spec.model_name }}
Generated by Aegis-ML - Production-Grade AI Reliability Engineer
"""

import os
import json
import argparse
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

{% if spec.method in ["lora", "qlora"] %}
from peft import LoraConfig, get_peft_model, TaskType
{% endif %}

{% if spec.method == "qlora" %}
from transformers import BitsAndBytesConfig
{% endif %}


@dataclass
class TrainingConfig:
    """Training configuration derived from spec."""
    model_name: str = "{{ spec.model_name }}"
    dataset_path: str = "{{ spec.dataset_path }}"
    num_epochs: int = {{ spec.num_epochs }}
    micro_batch_size: int = {{ spec.micro_batch_size }}
    gradient_accumulation_steps: int = {{ spec.gradient_accumulation_steps }}
    learning_rate: float = {{ spec.learning_rate }}
    seq_len: int = {{ spec.seq_len }}
    output_dir: str = "./output"


def load_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with appropriate configuration."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

{% if spec.method == "qlora" %}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
{% elif spec.method == "full_finetune" %}
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        trust_remote_code=True
    )
{% else %}
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        trust_remote_code=True
    )
{% endif %}

    return model, tokenizer


{% if spec.method in ["lora", "qlora"] %}
def setup_lora(model: AutoModelForCausalLM, config: TrainingConfig):
    """Configure LoRA adapters."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return get_peft_model(model, lora_config)
{% endif %}


def load_and_preprocess_data(config: TrainingConfig, tokenizer):
    """Load and preprocess dataset."""
    # Load JSONL dataset
    dataset = load_dataset("json", data_files=config.dataset_path, split="train")

    def tokenize_function(examples):
        # Assuming JSONL has "text" field
        texts = examples.get("text", examples.get("prompt", []))
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.seq_len,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset


def train(config: TrainingConfig):
    """Main training loop."""
    print(f"Starting training for {config.model_name}")
    print(f"Method: {{ '{{ spec.method }}' }}")
    print(f"Batch size: {config.micro_batch_size} x {config.gradient_accumulation_steps}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

{% if spec.method in ["lora", "qlora"] %}
    model = setup_lora(model, config)
    model.print_trainable_parameters()
{% endif %}

    # Load data
    train_dataset = load_and_preprocess_data(config, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=True,  # Use bfloat16 for A10G
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Train
    result = trainer.train()

    # Save
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Print final metrics
    print(f"Training complete!")
    print(f"Final loss: {result.training_loss:.4f}")

    return {
        "final_loss": result.training_loss,
        "total_steps": result.global_step,
        "model_path": final_model_path
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, help="JSON spec file")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.spec:
        with open(args.spec) as f:
            spec_data = json.load(f)
            for key, value in spec_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

**Step 4: Create the generator class**

```python
# aegis/generators/__init__.py
from aegis.generators.template import TemplateCodeGenerator

__all__ = ["TemplateCodeGenerator"]

# aegis/generators/base.py
from abc import ABC, abstractmethod
from aegis.models.state import TrainingSpec


class CodeGenerator(ABC):
    @abstractmethod
    def generate(self, spec: TrainingSpec) -> str:
        """Generate training script code from spec."""
        pass

# aegis/generators/template.py
import os
from jinja2 import Environment, FileSystemLoader
from aegis.generators.base import CodeGenerator
from aegis.models.state import TrainingSpec


class TemplateCodeGenerator(CodeGenerator):
    """Generate training scripts using Jinja2 templates."""

    def __init__(self, template_dir: str | None = None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template("hf_training.py.j2")

    def generate(self, spec: TrainingSpec) -> str:
        """Render training script from spec."""
        return self.template.render(spec=spec)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/generators/test_templates.py -v`
Expected: PASS (all 4 tests)

**Step 6: Commit**

```bash
git add templates/ aegis/generators/ tests/generators/
git commit -m "feat: add Jinja2 template-based code generator"
```

---

## Phase 3: LangGraph Nodes

### Task 3.1: Intent Parser Node

**Files:**
- Create: `aegis/nodes/__init__.py`
- Create: `aegis/nodes/intent.py`
- Create: `tests/nodes/test_intent.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_intent.py
import pytest
from aegis.models.state import AegisState, RunEvent, TrainingSpec
from aegis.nodes.intent import parse_intent_node


def test_parse_intent_creates_spec():
    state = AegisState()
    user_input = "I want to fine-tune tinyllama-272m on my data using LoRA"

    new_state = parse_intent_node(state, user_input=user_input)

    assert new_state.spec is not None
    assert isinstance(new_state.spec, TrainingSpec)
    assert new_state.spec.method == "lora"
    assert "tinyllama" in new_state.spec.model_name.lower()


def test_parse_intent_adds_event_log():
    state = AegisState()
    user_input = "Train tinyllama with lora"

    new_state = parse_intent_node(state, user_input=user_input)

    assert len(new_state.events) > 0
    assert new_state.events[0].phase == "parse_intent"
    assert new_state.events[0].status in ["completed", "failed"]


def test_parse_intent_returns_immutably():
    state = AegisState()
    user_input = "Train with LoRA"

    new_state = parse_intent_node(state, user_input=user_input)

    # Original state unchanged
    assert state.spec is None
    assert len(state.events) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_intent.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/__init__.py
from aegis.nodes.intent import parse_intent_node

__all__ = ["parse_intent_node"]

# aegis/nodes/intent.py
import re
import json
from aegis.models.state import AegisState, TrainingSpec, RunEvent


def parse_intent_node(state: AegisState, user_input: str) -> AegisState:
    """
    Parse user intent and extract TrainingSpec.
    MVP: Rule-based extraction (can be upgraded to LLM-based).
    """
    try:
        # Extract method
        method = "lora"  # default
        if "full" in user_input.lower() and "finetune" in user_input.lower():
            method = "full_finetune"
        elif "qlora" in user_input.lower() or "q-lora" in user_input.lower():
            method = "qlora"
        elif "lora" in user_input.lower():
            method = "lora"

        # Extract model name (look for patterns like "tinyllama", "llama-7b", etc.)
        model_map = {
            "tinyllama": "tinyllama/tinyllama-272m",
            "llama-7b": "meta-llama/Llama-2-7b-hf",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        }
        model_name = "tinyllama/tinyllama-272m"  # default
        for key, value in model_map.items():
            if key in user_input.lower():
                model_name = value
                break

        # Extract batch size if mentioned
        batch_match = re.search(r'batch\s*(?:size)?\s*(\d+)', user_input.lower())
        batch_size = int(batch_match.group(1)) if batch_match else 4

        # Extract epochs if mentioned
        epoch_match = re.search(r'(\d+)\s*epoch', user_input.lower())
        num_epochs = int(epoch_match.group(1)) if epoch_match else 3

        # Create spec
        spec = TrainingSpec(
            method=method,
            model_name=model_name,
            dataset_path="./data/sample.jsonl",  # default for MVP
            num_epochs=num_epochs,
            micro_batch_size=batch_size,
        )

        # Add event log
        event = RunEvent(
            phase="parse_intent",
            status="completed",
            message=f"Extracted {method} training spec for {model_name}",
            data={"parsed_spec": spec.model_dump()}
        )

        return state.model_copy(
            update={"spec": spec, "events": state.events + [event]}
        )

    except Exception as e:
        # Log failure
        event = RunEvent(
            phase="parse_intent",
            status="failed",
            message=f"Failed to parse intent: {str(e)}"
        )
        return state.model_copy(update={"events": state.events + [event]})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_intent.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/ tests/nodes/
git commit -m "feat: add intent parser node with rule-based extraction"
```

---

### Task 3.2: Cost Estimator Node

**Files:**
- Create: `aegis/nodes/profiler.py`
- Create: `tests/nodes/test_profiler.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_profiler.py
import pytest
from aegis.models.state import AegisState, TrainingSpec, RunEvent
from aegis.nodes.profiler import estimate_cost_node


def test_estimate_cost_adds_cost_estimate():
    state = AegisState(spec=TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        target_gpu="a10g"
    ))

    new_state = estimate_cost_node(state)

    assert new_state.cost_estimate is not None
    assert new_state.cost_estimate.estimated_cost_usd > 0


def test_estimate_cost_adds_event_log():
    state = AegisState(spec=TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl"
    ))

    new_state = estimate_cost_node(state)

    assert any(e.phase == "estimate_cost" for e in new_state.events)


def test_estimate_cost_requires_spec():
    state = AegisState(spec=None)

    new_state = estimate_cost_node(state)

    # Should add a failure event
    assert any(
        e.phase == "estimate_cost" and e.status == "failed"
        for e in new_state.events
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_profiler.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes.profiler'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/profiler.py
from aegis.models.state import AegisState, RunEvent
from aegis.profilers.cost import estimate_training_cost


def estimate_cost_node(state: AegisState) -> AegisState:
    """Estimate training cost and VRAM requirements."""
    if state.spec is None:
        event = RunEvent(
            phase="estimate_cost",
            status="failed",
            message="Cannot estimate cost: no training spec provided"
        )
        return state.model_copy(update={"events": state.events + [event]})

    try:
        cost_estimate = estimate_training_cost(state.spec)

        event = RunEvent(
            phase="estimate_cost",
            status="completed",
            message=f"Estimated ${cost_estimate.estimated_cost_usd:.4f}, "
                   f"{cost_estimate.estimated_vram_gb:.1f}GB VRAM",
            data=cost_estimate.model_dump()
        )

        return state.model_copy(
            update={
                "cost_estimate": cost_estimate,
                "events": state.events + [event]
            }
        )
    except Exception as e:
        event = RunEvent(
            phase="estimate_cost",
            status="failed",
            message=f"Cost estimation failed: {str(e)}"
        )
        return state.model_copy(update={"events": state.events + [event]})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_profiler.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/profiler.py tests/nodes/test_profiler.py
git commit -m "feat: add cost estimator node"
```

---

### Task 3.3: Budget Gate Node (HITL with interrupt)

**Files:**
- Create: `aegis/nodes/gate.py`
- Create: `tests/nodes/test_gate.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_gate.py
import pytest
from aegis.models.state import AegisState, TrainingSpec, CostEstimate, RunEvent
from aegis.nodes.gate import budget_gate_node, check_auto_approve


def test_check_auto_approve_under_threshold():
    state = AegisState(
        budget_policy=AegisState().budget_policy.model_copy(update={"soft_threshold_usd": 2.0}),
        cost_estimate=CostEstimate(
            estimated_cost_usd=1.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        )
    )

    approved = check_auto_approve(state)
    assert approved is True


def test_check_auto_approve_over_soft_threshold():
    state = AegisState(
        budget_policy=AegisState().budget_policy.model_copy(update={"soft_threshold_usd": 2.0}),
        cost_estimate=CostEstimate(
            estimated_cost_usd=3.0,  # Over soft threshold
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        )
    )

    approved = check_auto_approve(state)
    assert approved is False


def test_check_auto_approve_exceeds_hard_limit():
    state = AegisState(
        budget_policy=AegisState().budget_policy.model_copy(update={"max_budget_usd": 5.0}),
        cost_estimate=CostEstimate(
            estimated_cost_usd=6.0,  # Over hard limit
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        )
    )

    approved = check_auto_approve(state)
    assert approved is False


def test_budget_gate_adds_event():
    state = AegisState(
        cost_estimate=CostEstimate(
            estimated_cost_usd=1.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        )
    )

    new_state = budget_gate_node(state)

    assert any(e.phase == "budget_gate" for e in new_state.events)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_gate.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes.gate'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/gate.py
from aegis.models.state import AegisState, RunEvent


def check_auto_approve(state: AegisState) -> bool:
    """
    Check if cost is under auto-approval threshold.
    Returns True if auto-approve, False if HITL needed.
    """
    if state.cost_estimate is None:
        return False

    if state.cost_estimate.estimated_cost_usd > state.budget_policy.max_budget_usd:
        return False

    if state.cost_estimate.estimated_cost_usd <= state.budget_policy.soft_threshold_usd:
        return True

    return False


def budget_gate_node(state: AegisState) -> AegisState:
    """
    Budget gate node that evaluates cost against policy.
    In full implementation, this uses interrupt() for HITL.
    For MVP, returns state with decision recorded.
    """
    if state.cost_estimate is None:
        event = RunEvent(
            phase="budget_gate",
            status="failed",
            message="Cannot evaluate budget: no cost estimate available"
        )
        return state.model_copy(update={"events": state.events + [event]})

    auto_approve = check_auto_approve(state)

    if state.cost_estimate.estimated_cost_usd > state.budget_policy.max_budget_usd:
        status_msg = f"Cost ${state.cost_estimate.estimated_cost_usd:.4f} exceeds maximum ${state.budget_policy.max_budget_usd:.2f}"
        status = "failed"
        decision = "cancel"
    elif auto_approve:
        status_msg = f"Cost ${state.cost_estimate.estimated_cost_usd:.4f} auto-approved"
        status = "completed"
        decision = "approve"
    else:
        status_msg = f"Cost ${state.cost_estimate.estimated_cost_usd:.4f} requires human approval"
        status = "interrupted"
        decision = None  # Will be set by HITL

    event = RunEvent(
        phase="budget_gate",
        status=status,
        message=status_msg,
        data={
            "auto_approve": auto_approve,
            "cost_usd": state.cost_estimate.estimated_cost_usd,
            "soft_threshold": state.budget_policy.soft_threshold_usd,
            "max_budget": state.budget_policy.max_budget_usd
        }
    )

    updates = {"events": state.events + [event]}
    if decision:
        updates["human_decision"] = decision

    return state.model_copy(update=updates)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_gate.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/gate.py tests/nodes/test_gate.py
git commit -m "feat: add budget gate node with auto-approval logic"
```

---

### Task 3.4: Code Generator Node

**Files:**
- Create: `aegis/nodes/generator.py`
- Create: `tests/nodes/test_generator.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_generator.py
import pytest
from aegis.models.state import AegisState, TrainingSpec, RunEvent
from aegis.nodes.generator import generate_code_node


def test_generate_code_adds_generated_code():
    state = AegisState(spec=TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl"
    ))

    new_state = generate_code_node(state)

    assert new_state.generated_code is not None
    assert isinstance(new_state.generated_code, str)
    assert len(new_state.generated_code) > 0


def test_generate_code_includes_imports():
    state = AegisState(spec=TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl"
    ))

    new_state = generate_code_node(state)

    assert "from transformers import" in new_state.generated_code


def test_generate_code_requires_spec():
    state = AegisState(spec=None)

    new_state = generate_code_node(state)

    assert new_state.generated_code is None
    assert any(
        e.phase == "generate_code" and e.status == "failed"
        for e in new_state.events
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_generator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes.generator'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/generator.py
from aegis.models.state import AegisState, RunEvent
from aegis.generators.template import TemplateCodeGenerator


def generate_code_node(state: AegisState) -> AegisState:
    """Generate training script from spec."""
    if state.spec is None:
        event = RunEvent(
            phase="generate_code",
            status="failed",
            message="Cannot generate code: no training spec provided"
        )
        return state.model_copy(update={"events": state.events + [event]})

    try:
        generator = TemplateCodeGenerator()
        code = generator.generate(state.spec)

        event = RunEvent(
            phase="generate_code",
            status="completed",
            message=f"Generated {len(code)} bytes of training code",
            data={"method": state.spec.method, "code_length": len(code)}
        )

        return state.model_copy(
            update={
                "generated_code": code,
                "events": state.events + [event]
            }
        )
    except Exception as e:
        event = RunEvent(
            phase="generate_code",
            status="failed",
            message=f"Code generation failed: {str(e)}"
        )
        return state.model_copy(update={"events": state.events + [event]})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_generator.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/generator.py tests/nodes/test_generator.py
git commit -m "feat: add code generator node"
```

---

### Task 3.5: Remediation Node

**Files:**
- Create: `aegis/nodes/remediate.py`
- Create: `tests/nodes/test_remediate.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_remediate.py
import pytest
from aegis.models.state import AegisState, TrainingSpec, RunEvent
from aegis.nodes.remediate import remediate_spec_node, remediate_spec


def test_remediate_oom_shrinks_batch_size():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=8
    )

    new_spec = remediate_spec(spec, "CUDA out of memory")

    assert new_spec.micro_batch_size == 4  # Halved


def test_reminate_oom_at_batch_1_switches_to_qlora():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=1
    )

    new_spec = remediate_spec(spec, "CUDA out of memory")

    assert new_spec.method == "qlora"


def test_reminate_full_finetune_at_batch_1_switches_to_lora():
    spec = TrainingSpec(
        method="full_finetune",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        micro_batch_size=1
    )

    new_spec = remediate_spec(spec, "CUDA out of memory")

    assert new_spec.method == "lora"


def test_reminate_nan_reduces_lr():
    spec = TrainingSpec(
        method="lora",
        model_name="tinyllama/tinyllama-272m",
        dataset_path="./data/sample.jsonl",
        learning_rate=5e-5
    )

    new_spec = remediate_spec(spec, "Loss became NaN")

    assert new_spec.learning_rate == 2.5e-5  # Halved


def test_remediate_node_updates_state():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl",
            micro_batch_size=8
        ),
        retry_count=0
    )

    new_state = remediate_spec_node(
        state,
        error_message="CUDA out of memory"
    )

    assert new_state.spec.micro_batch_size == 4
    assert new_state.retry_count == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_remediate.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes.remediate'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/remediate.py
from aegis.models.state import AegisState, TrainingSpec, RunEvent


def remediate_spec(spec: TrainingSpec, error_message: str) -> TrainingSpec:
    """
    Mutate TrainingSpec to fix common issues.
    Priority order (least invasive first):
    1. OOM → shrink batch size
    2. Batch at 1 → switch to more efficient method
    3. NaN → reduce learning rate
    4. Generic → shrink batch
    """
    error_lower = error_message.lower()

    # 1. Out of memory → shrink batch size
    if "out of memory" in error_lower or "cuda" in error_lower:
        new_batch = max(1, spec.micro_batch_size // 2)
        if new_batch < spec.micro_batch_size:
            return spec.model_copy(update={"micro_batch_size": new_batch})

    # 2. If batch at 1 → switch method
    if spec.micro_batch_size == 1:
        if spec.method == "full_finetune":
            return spec.model_copy(update={"method": "lora"})
        elif spec.method == "lora":
            return spec.model_copy(update={"method": "qlora"})

    # 3. NaN or loss spike → reduce learning rate
    if "nan" in error_lower or "loss spike" in error_lower:
        return spec.model_copy(update={"learning_rate": spec.learning_rate / 2})

    # 4. Generic batch reduction
    new_batch = max(1, spec.micro_batch_size // 2)
    return spec.model_copy(update={"micro_batch_size": new_batch})


def remediate_spec_node(state: AegisState, error_message: str) -> AegisState:
    """Apply remediation and update retry count."""
    if state.spec is None:
        event = RunEvent(
            phase="remediate_spec",
            status="failed",
            message="Cannot remediate: no training spec"
        )
        return state.model_copy(update={"events": state.events + [event]})

    new_spec = remediate_spec(state.spec, error_message)

    # Detect what changed
    changes = []
    if new_spec.micro_batch_size != state.spec.micro_batch_size:
        changes.append(f"batch_size: {state.spec.micro_batch_size} → {new_spec.micro_batch_size}")
    if new_spec.method != state.spec.method:
        changes.append(f"method: {state.spec.method} → {new_spec.method}")
    if new_spec.learning_rate != state.spec.learning_rate:
        changes.append(f"lr: {state.spec.learning_rate} → {new_spec.learning_rate}")

    change_desc = ", ".join(changes) if changes else "no changes"

    event = RunEvent(
        phase="remediate_spec",
        status="completed",
        message=f"Applied remediation: {change_desc}",
        data={
            "error": error_message,
            "changes": changes,
            "new_spec": new_spec.model_dump()
        }
    )

    return state.model_copy(
        update={
            "spec": new_spec,
            "events": state.events + [event],
            "retry_count": state.retry_count + 1
        }
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_remediate.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/remediate.py tests/nodes/test_remediate.py
git commit -m "feat: add remediation node with intelligent spec mutation"
```

---

### Task 3.6: Evaluator Node

**Files:**
- Create: `aegis/nodes/evaluator.py`
- Create: `tests/nodes/test_evaluator.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_evaluator.py
import pytest
from aegis.models.state import AegisState, RunEvent
from aegis.nodes.evaluator import run_evals_node, run_evals


def test_run_evals_passing():
    execution_result = {
        "metrics": {
            "train_loss": 1.5,
            "eval_loss": 1.8
        },
        "model_path": "/tmp/model"
    }

    result = run_evals(execution_result)

    assert result["passed"] is True
    assert len(result["failures"]) == 0


def test_run_evals_high_loss_fails():
    execution_result = {
        "metrics": {
            "train_loss": 2.5,  # > 2.0 threshold
        },
        "model_path": "/tmp/model"
    }

    result = run_evals(execution_result)

    assert result["passed"] is False
    assert "Loss did not converge" in result["failures"][0]


def test_run_evals_overfitting_fails():
    execution_result = {
        "metrics": {
            "train_loss": 0.5,
            "eval_loss": 2.0  # Gap > 1.0
        },
        "model_path": "/tmp/model"
    }

    result = run_evals(execution_result)

    assert result["passed"] is False
    assert "overfitting" in result["failures"][0].lower()


def test_run_evals_node_adds_to_state():
    state = AegisState(
        execution_result={
            "metrics": {"train_loss": 1.5},
            "model_path": "/tmp/model"
        }
    )

    new_state = run_evals_node(state)

    assert new_state.eval_result is not None
    assert new_state.eval_result["passed"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_evaluator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes.evaluator'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/evaluator.py
from aegis.models.state import AegisState, RunEvent


def run_evals(execution_result: dict) -> dict:
    """
    Post-training evaluation checks.
    Returns: {"passed": bool, "metrics": dict, "failures": list}
    """
    metrics = execution_result.get("metrics", {})
    failures = []

    # Gate 1: Loss convergence check
    final_loss = metrics.get("train_loss", float("inf"))
    if final_loss > 2.0:
        failures.append(f"Loss did not converge: {final_loss:.3f} > 2.0")

    # Gate 2: Overfitting check
    if "eval_loss" in metrics:
        train_eval_gap = metrics["eval_loss"] - metrics["train_loss"]
        if train_eval_gap > 1.0:
            failures.append(f"Severe overfitting: {train_eval_gap:.2f} gap")

    # Gate 3: Canary safety test (stub for MVP)
    canary_passed = _run_canary_test(execution_result.get("model_path"))
    if not canary_passed:
        failures.append("Canary safety test failed")

    return {
        "passed": len(failures) == 0,
        "metrics": metrics,
        "failures": failures
    }


def _run_canary_test(model_path: str | None) -> bool:
    """Safety probe - MVP always passes."""
    return True


def run_evals_node(state: AegisState) -> AegisState:
    """Run evaluation checks and update state."""
    if state.execution_result is None:
        event = RunEvent(
            phase="run_evals",
            status="failed",
            message="Cannot run evals: no execution result"
        )
        return state.model_copy(update={"events": state.events + [event]})

    try:
        eval_result = run_evals(state.execution_result)

        status = "completed" if eval_result["passed"] else "failed"
        message = f"Evaluation {'passed' if eval_result['passed'] else 'failed'}"
        if eval_result["failures"]:
            message += f": {len(eval_result['failures'])} failures"

        event = RunEvent(
            phase="run_evals",
            status=status,
            message=message,
            data=eval_result
        )

        return state.model_copy(
            update={
                "eval_result": eval_result,
                "events": state.events + [event]
            }
        )
    except Exception as e:
        event = RunEvent(
            phase="run_evals",
            status="failed",
            message=f"Eval failed: {str(e)}"
        )
        return state.model_copy(update={"events": state.events + [event]})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_evaluator.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/evaluator.py tests/nodes/test_evaluator.py
git commit -m "feat: add evaluator node with loss and overfitting checks"
```

---

### Task 3.7: Report Generator Node

**Files:**
- Create: `aegis/nodes/reporter.py`
- Create: `tests/nodes/test_reporter.py`

**Step 1: Write the failing test**

```python
# tests/nodes/test_reporter.py
import pytest
from aegis.models.state import AegisState, TrainingSpec, CostEstimate, BudgetPolicy, RunEvent
from aegis.nodes.reporter import write_report_node, generate_report


def test_generate_report_creates_markdown():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl"
        ),
        budget_policy=BudgetPolicy(max_budget_usd=5.0),
        cost_estimate=CostEstimate(
            estimated_cost_usd=0.0150,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        ),
        events=[
            RunEvent(phase="parse_intent", status="completed", message="Parsed"),
            RunEvent(phase="estimate_cost", status="completed", message="Estimated")
        ]
    )

    report = generate_report(state)

    assert "# " in report  # Markdown header
    assert "Aegis-ML" in report
    assert "tinyllama" in report
    assert "lora" in report
    assert "$" in report  # Cost info
    assert "| Time | Phase |" in report  # Event table


def test_generate_report_includes_executive_summary():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl"
        ),
        cost_estimate=CostEstimate(
            estimated_cost_usd=0.0150,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        )
    )

    report = generate_report(state)

    assert "## Executive Summary" in report
    assert "| Metric | Value |" in report


def test_write_report_node_updates_state():
    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl"
        )
    )

    new_state = write_report_node(state)

    assert new_state.final_report is not None
    assert len(new_state.final_report) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/nodes/test_reporter.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.nodes.reporter'"

**Step 3: Write minimal implementation**

```python
# aegis/nodes/reporter.py
from aegis.models.state import AegisState, RunEvent
from datetime import datetime


def generate_report(state: AegisState) -> str:
    """
    Generate markdown report proving 'Adult in the Room' narrative.
    This is what hiring managers read.
    """
    model_name = state.spec.model_name if state.spec else "unknown"
    timestamp_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    report = f"""# 🛡️ Aegis-ML Training Report

**Run ID:** {model_name.split('/')[-1]}-{timestamp_str}
**Generated:** {datetime.utcnow().isoformat()}

---

## Executive Summary

| Metric | Value |
|--------|-------|
"""

    # Add spec info
    if state.spec:
        report += f"| Model | {state.spec.model_name} |\n"
        report += f"| Method | {state.spec.method} |\n"

    # Add cost info
    if state.cost_estimate:
        report += f"| Estimated Cost | ${state.cost_estimate.estimated_cost_usd:.4f} |\n"
        report += f"| Est. VRAM | {state.cost_estimate.estimated_vram_gb:.1f} GB |\n"
        report += f"| Est. Duration | {state.cost_estimate.estimated_duration_min:.1f} min |\n"

    # Add execution info
    if state.execution_result:
        duration = state.execution_result.get("duration_sec", 0) / 60
        report += f"| Actual Duration | {duration:.1f} min |\n"

    # Add eval status
    if state.eval_result:
        status_icon = "✅ PASSED" if state.eval_result.get("passed") else "❌ FAILED"
        report += f"| Status | {status_icon} |\n"

    report += "\n---\n\n## Training Specification\n\n"

    if state.spec:
        report += f"```json\n{state.spec.model_dump_json(indent=2)}\n```\n"

    # Cost gate section
    report += "\n---\n\n## Cost Gate Decision\n\n"
    report += "| Threshold | Value |\n|-----------|-------|\n"
    report += f"| Soft Limit | ${state.budget_policy.soft_threshold_usd:.2f} |\n"
    report += f"| Hard Limit | ${state.budget_policy.max_budget_usd:.2f} |\n"

    if state.cost_estimate:
        report += f"| Estimated | ${state.cost_estimate.estimated_cost_usd:.4f} |\n"

        if state.cost_estimate.estimated_cost_usd < state.budget_policy.soft_threshold_usd:
            decision = "AUTO-APPROVED"
        elif state.human_decision:
            decision = f"HUMAN-APPROVED ({state.human_decision})"
        else:
            decision = "PENDING"
        report += f"| Decision | {decision} |\n"

    # Event timeline
    report += "\n---\n\n## Event Timeline\n\n"
    report += "| Time | Phase | Status | Message |\n"
    report += "|------|-------|--------|---------|\n"

    for event in state.events:
        time_str = event.timestamp.strftime("%H:%M:%S")
        report += f"| {time_str} | {event.phase} | {event.status} | {event.message} |\n"

    # Eval failures
    if state.eval_result and state.eval_result.get("failures"):
        report += "\n---\n\n## ⚠️ Evaluation Failures\n\n"
        for failure in state.eval_result["failures"]:
            report += f"- {failure}\n"

    report += "\n---\n\n*Generated by Aegis-ML - Production-Grade AI Reliability Engineer*"

    return report


def write_report_node(state: AegisState) -> AegisState:
    """Generate report and add to state."""
    try:
        report = generate_report(state)

        event = RunEvent(
            phase="write_report",
            status="completed",
            message=f"Generated report ({len(report)} chars)"
        )

        return state.model_copy(
            update={
                "final_report": report,
                "events": state.events + [event]
            }
        )
    except Exception as e:
        event = RunEvent(
            phase="write_report",
            status="failed",
            message=f"Report generation failed: {str(e)}"
        )
        return state.model_copy(update={"events": state.events + [event]})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/nodes/test_reporter.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add aegis/nodes/reporter.py tests/nodes/test_reporter.py
git commit -m "feat: add report generator node for hiring manager artifacts"
```

---

## Phase 4: Conditional Edges

### Task 4.1: Budget Gate Routing

**Files:**
- Create: `aegis/edges/__init__.py`
- Create: `aegis/edges/budget.py`
- Create: `tests/edges/test_budget.py`

**Step 1: Write the failing test**

```python
# tests/edges/test_budget.py
import pytest
from aegis.models.state import AegisState, TrainingSpec, CostEstimate, BudgetPolicy, RunEvent
from aegis.edges.budget import budget_gate_routing, budget_decision_routing


def test_budget_gate_routing_no_estimate():
    state = AegisState(cost_estimate=None)
    result = budget_gate_routing(state)
    assert result == "budget_gate"  # Should route to gate which will handle error


def test_budget_gate_routing_exceeds_max():
    state = AegisState(
        cost_estimate=CostEstimate(
            estimated_cost_usd=10.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=10.0,
            cost_breakdown={}
        ),
        budget_policy=BudgetPolicy(max_budget_usd=5.0)
    )
    result = budget_gate_routing(state)
    assert result == "budget_gate"  # Route to HITL


def test_budget_gate_routing_auto_approve():
    state = AegisState(
        cost_estimate=CostEstimate(
            estimated_cost_usd=1.0,
            estimated_vram_gb=5.0,
            estimated_duration_min=5.0,
            cost_breakdown={}
        ),
        budget_policy=BudgetPolicy(soft_threshold_usd=2.0)
    )
    result = budget_gate_routing(state)
    assert result == "budget_gate"  # Always route through gate node


def test_budget_decision_routing_approve():
    state = AegisState(human_decision="approve")
    result = budget_decision_routing(state)
    assert result == "generate_code"


def test_budget_decision_routing_optimize():
    state = AegisState(human_decision="optimize")
    result = budget_decision_routing(state)
    assert result == "remediate_spec"


def test_budget_decision_routing_cancel():
    state = AegisState(human_decision="cancel")
    result = budget_decision_routing(state)
    assert result == "__end__"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/edges/test_budget.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.edges'"

**Step 3: Write minimal implementation**

```python
# aegis/edges/__init__.py
from aegis.edges.budget import budget_gate_routing, budget_decision_routing

__all__ = ["budget_gate_routing", "budget_decision_routing"]

# aegis/edges/budget.py
from aegis.models.state import AegisState


def budget_gate_routing(state: AegisState) -> str:
    """
    Route from estimate_cost to appropriate next step.
    For MVP: always routes through budget_gate node.
    The gate node itself handles auto-approve vs HITL.
    """
    return "budget_gate"


def budget_decision_routing(state: AegisState) -> str:
    """
    Route from budget_gate after human decision.
    """
    if state.human_decision == "approve":
        return "generate_code"
    elif state.human_decision == "optimize":
        return "remediate_spec"
    elif state.human_decision == "cancel":
        return "__end__"
    else:
        # No decision yet - should have been set by gate node
        return "__end__"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/edges/test_budget.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add aegis/edges/ tests/edges/
git commit -m "feat: add budget gate conditional edge routing"
```

---

### Task 4.2: Retry Routing

**Files:**
- Create: `aegis/edges/retry.py`
- Create: `tests/edges/test_retry.py`

**Step 1: Write the failing test**

```python
# tests/edges/test_retry.py
import pytest
from aegis.models.state import AegisState
from aegis.edges.retry import retry_routing


def test_retry_routing_has_retries():
    state = AegisState(retry_count=1, max_retries=3)
    result = retry_routing(state)
    assert result == "remediate_spec"


def test_retry_routing_exhausted():
    state = AegisState(retry_count=3, max_retries=3)
    result = retry_routing(state)
    assert result == "write_report"  # End with report


def test_retry_routing_at_zero():
    state = AegisState(retry_count=0, max_retries=3)
    result = retry_routing(state)
    assert result == "remediate_spec"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/edges/test_retry.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.edges.retry'"

**Step 3: Write minimal implementation**

```python
# aegis/edges/retry.py
from aegis.models.state import AegisState


def retry_routing(state: AegisState) -> str:
    """
    Route after execution failure.
    If retries remaining → remediate_spec
    If exhausted → write_report (then END)
    """
    if state.retry_count < state.max_retries:
        return "remediate_spec"
    else:
        return "write_report"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/edges/test_retry.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add aegis/edges/retry.py tests/edges/test_retry.py
git commit -m "feat: add retry conditional edge routing"
```

---

### Task 4.3: Eval Gate Routing

**Files:**
- Create: `aegis/edges/eval.py`
- Create: `tests/edges/test_eval.py`

**Step 1: Write the failing test**

```python
# tests/edges/test_eval.py
import pytest
from aegis.models.state import AegisState
from aegis.edges.eval import eval_gate_routing


def test_eval_gate_routing_passed():
    state = AegisState(eval_result={"passed": True})
    result = eval_gate_routing(state)
    assert result == "write_report"


def test_eval_gate_routing_failed():
    state = AegisState(eval_result={"passed": False, "failures": ["Loss too high"]})
    result = eval_gate_routing(state)
    assert result == "remediate_spec"


def test_eval_gate_routing_no_result():
    state = AegisState(eval_result=None)
    result = eval_gate_routing(state)
    assert result == "write_report"  # End without eval
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/edges/test_eval.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.edges.eval'"

**Step 3: Write minimal implementation**

```python
# aegis/edges/eval.py
from aegis.models.state import AegisState


def eval_gate_routing(state: AegisState) -> str:
    """
    Route after evaluation.
    If passed → write_report
    If failed → remediate_spec
    If no eval result → write_report (end)
    """
    if state.eval_result is None:
        return "write_report"

    if state.eval_result.get("passed", False):
        return "write_report"
    else:
        return "remediate_spec"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/edges/test_eval.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add aegis/edges/eval.py tests/edges/test_eval.py
git commit -m "feat: add evaluation gate conditional edge routing"
```

---

## Phase 5: Graph Construction

### Task 5.1: Main LangGraph

**Files:**
- Create: `aegis/graph.py`
- Create: `tests/test_graph.py`

**Step 1: Write the failing test**

```python
# tests/test_graph.py
import pytest
from aegis.graph import build_aegis_graph
from aegis.models.state import AegisState


def test_graph_builds():
    graph = build_aegis_graph()
    assert graph is not None


def test_graph_has_all_nodes():
    graph = build_aegis_graph()
    # Graph should have nodes defined
    nodes = graph.nodes
    expected_nodes = [
        "parse_intent",
        "estimate_cost",
        "budget_gate",
        "generate_code",
        "execute",
        "run_evals",
        "eval_gate",
        "remediate_spec",
        "check_retries",
        "write_report"
    ]
    for node in expected_nodes:
        assert node in nodes


def test_graph_entry_point():
    graph = build_aegis_graph()
    # Entry point should be parse_intent
    assert graph.get_graph().first_node() == "parse_intent"


def test_graph_invoke_simple_flow():
    graph = build_aegis_graph()
    initial_state = AegisState()

    # Invoke with user input
    config = {"configurable": {"thread_id": "test-thread"}}
    result = graph.invoke(
        initial_state,
        config=config
    )

    assert result is not None
    # Should have parsed spec
    assert result.spec is not None
    # Should have cost estimate
    assert result.cost_estimate is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.graph'"

**Step 3: Write minimal implementation**

```python
# aegis/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from aegis.models.state import AegisState
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


# Stub executor node for MVP (will be implemented later)
def execute_node(state: AegisState) -> AegisState:
    """Stub executor - returns mock result."""
    from aegis.models.state import RunEvent

    if state.generated_code is None:
        event = RunEvent(
            phase="execute",
            status="failed",
            message="No code to execute"
        )
        return state.model_copy(update={"events": state.events + [event]})

    # Mock successful execution
    mock_result = {
        "metrics": {"train_loss": 1.2, "eval_loss": 1.5},
        "model_path": "/tmp/mock_model",
        "duration_sec": 300
    }

    event = RunEvent(
        phase="execute",
        status="completed",
        message="Mock execution completed",
        data=mock_result
    )

    return state.model_copy(
        update={
            "execution_result": mock_result,
            "events": state.events + [event]
        }
    )


# Stub eval gate node (routing is handled by edge)
def eval_gate_node(state: AegisState) -> AegisState:
    """Pass-through node for evaluation gate."""
    return state


# Stub check_retries node (routing is handled by edge)
def check_retries_node(state: AegisState) -> AegisState:
    """Pass-through node for retry checking."""
    return state


def build_aegis_graph() -> StateGraph:
    """Build and compile the Aegis LangGraph."""
    builder = StateGraph(AegisState)

    # Add all nodes
    builder.add_node("parse_intent", parse_intent_node)
    builder.add_node("estimate_cost", estimate_cost_node)
    builder.add_node("budget_gate", budget_gate_node)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("execute", execute_node)
    builder.add_node("run_evals", run_evals_node)
    builder.add_node("eval_gate", eval_gate_node)
    builder.add_node("remediate_spec", remediate_spec_node)
    builder.add_node("check_retries", check_retries_node)
    builder.add_node("write_report", write_report_node)

    # Define edges
    builder.set_entry_point("parse_intent")

    builder.add_edge("parse_intent", "estimate_cost")
    builder.add_conditional_edges("estimate_cost", budget_gate_routing)
    builder.add_conditional_edges("budget_gate", budget_decision_routing)
    builder.add_edge("generate_code", "execute")
    builder.add_edge("execute", "run_evals")
    builder.add_edge("run_evals", "eval_gate")
    builder.add_conditional_edges("eval_gate", eval_gate_routing)
    builder.add_conditional_edges("check_retries", retry_routing)

    # Remediation loops back to estimate_cost
    builder.add_edge("remediate_spec", "estimate_cost")

    # Write report ends the graph
    builder.add_edge("write_report", END)

    # Compile with memory for persistence
    memory = SqliteSaver.from_conn_string("aegis_checkpoints.db")

    return builder.compile(checkpointer=memory)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add aegis/graph.py tests/test_graph.py
git commit -m "feat: build complete LangGraph with all nodes and edges"
```

---

## Phase 6: Streamlit UI

### Task 6.1: Basic Streamlit App

**Files:**
- Create: `ui/streamlit_app.py`
- Create: `tests/test_ui.py`

**Step 1: Write the failing test**

```python
# tests/test_ui.py
import pytest


def test_ui_file_exists():
    import os
    assert os.path.exists("ui/streamlit_app.py")


def test_ui_imports_work():
    import sys
    sys.path.insert(0, ".")
    # Just verify imports don't crash
    from aegis.models.state import AegisState
    from aegis.graph import build_aegis_graph
    assert AegisState is not None
    assert build_aegis_graph is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ui.py -v`
Expected: FAIL with "AssertionError: assert False" (file doesn't exist yet)

**Step 3: Write minimal implementation**

```python
# ui/streamlit_app.py
"""
Aegis-ML Streamlit UI
Production-Grade AI Reliability Engineer
"""

import streamlit as st
from aegis.models.state import AegisState, RunEvent
from aegis.graph import build_aegis_graph

st.set_page_config(
    page_title="Aegis-ML",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = build_aegis_graph()
if "aegis_state" not in st.session_state:
    st.session_state.aegis_state = AegisState()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-thread"

# Sidebar: Status and metrics
with st.sidebar:
    st.title("🛡️ Aegis-ML")
    st.markdown("### Production-Grade AI Reliability Engineer")

    st.markdown("---")

    # Cost gate status
    if st.session_state.aegis_state.cost_estimate:
        st.metric(
            "Est. Cost",
            f"${st.session_state.aegis_state.cost_estimate.estimated_cost_usd:.4f}"
        )
        st.metric(
            "VRAM",
            f"{st.session_state.aegis_state.cost_estimate.estimated_vram_gb:.1f} GB"
        )
    else:
        st.caption("No cost estimate yet")

    st.markdown("---")

    # Phase tracker
    st.markdown("### Pipeline Phase")

    # Determine current phase from events
    phases = ["parse_intent", "estimate_cost", "budget_gate", "generate_code", "execute", "eval"]
    events = st.session_state.aegis_state.events

    current_phase_idx = 0
    for i, phase in enumerate(phases):
        if any(e.phase == phase and e.status == "completed" for e in events):
            st.success(phase)
            current_phase_idx = i + 1
        elif any(e.phase == phase for e in events):
            st.info(f"➤ {phase}")
            current_phase_idx = i
        else:
            st.caption(phase)

    st.markdown("---")

    # Event log
    st.markdown("### Event Log")
    for event in reversed(st.session_state.aegis_state.events[-10:]):
        status_icon = {
            "completed": "✅",
            "failed": "❌",
            "started": "▶️",
            "interrupted": "⏸️"
        }.get(event.status, "•")

        st.caption(f"{status_icon} {event.phase}: {event.message}")

    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset"):
        st.session_state.aegis_state = AegisState()
        st.rerun()

# Main content area
st.title("Fine-Tuning Orchestrator")
st.markdown("Describe your fine-tuning job and Aegis will handle the rest safely.")

# Chat input for user intent
user_input = st.chat_input("e.g., 'I want to fine-tune tinyllama with LoRA for 3 epochs'")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # Invoke graph with user input
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = st.session_state.graph.invoke(
        st.session_state.aegis_state,
        config=config
    )

    st.session_state.aegis_state = result
    st.rerun()

# Show current spec if available
if st.session_state.aegis_state.spec:
    st.markdown("---")
    st.subheader("📋 Training Specification")

    spec = st.session_state.aegis_state.spec
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

# Show generated code if available
if st.session_state.aegis_state.generated_code:
    st.markdown("---")
    st.subheader("🐍 Generated Training Script")

    with st.expander("View code", expanded=False):
        st.code(st.session_state.aegis_state.generated_code, language="python")

# Show execution result if available
if st.session_state.aegis_state.execution_result:
    st.markdown("---")
    st.subheader("📊 Execution Results")

    result = st.session_state.aegis_state.execution_result
    col1, col2 = st.columns(2)

    with col1:
        if "metrics" in result:
            for key, value in result["metrics"].items():
                st.metric(key, f"{value:.4f}")

    with col2:
        if "duration_sec" in result:
            st.metric("Duration", f"{result['duration_sec'] / 60:.1f} min")

# Show eval result if available
if st.session_state.aegis_state.eval_result:
    st.markdown("---")
    st.subheader("🔍 Evaluation Results")

    eval_result = st.session_state.aegis_state.eval_result

    if eval_result.get("passed"):
        st.success("✅ All evaluations passed!")
    else:
        st.error("❌ Some evaluations failed")

        if eval_result.get("failures"):
            for failure in eval_result["failures"]:
                st.warning(f"- {failure}")

# Show final report if available
if st.session_state.aegis_state.final_report:
    st.markdown("---")
    st.subheader("📄 Final Report")

    st.markdown(st.session_state.aegis_state.final_report)

    # Download button
    st.download_button(
        label="📥 Download Report",
        data=st.session_state.aegis_state.final_report,
        file_name="aegis_report.md",
        mime="text/markdown"
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ui.py -v`
Expected: PASS (all 2 tests)

**Step 5: Commit**

```bash
git add ui/ tests/test_ui.py
git commit -m "feat: add Streamlit UI with chat, status sidebar, and report viewer"
```

---

## Phase 7: Modal Integration

### Task 7.1: Modal Executor

**Files:**
- Create: `aegis/executors/__init__.py`
- Create: `aegis/executors/modal_runner.py`
- Create: `tests/executors/test_modal.py`

**Step 1: Write the failing test**

```python
# tests/executors/test_modal.py
import pytest
from aegis.executors.modal_runner import ModalExecutor


def test_modal_executor_imports():
    # Verify modal can be imported (may not be configured)
    try:
        import modal
        assert True
    except ImportError:
        pytest.skip("modal not installed")


def test_modal_executor_can_be_created():
    executor = ModalExecutor()
    assert executor is not None


def test_modal_executor_generate_stub():
    executor = ModalExecutor()
    spec_code = executor._generate_stub()
    assert "modal" in spec_code.lower()
    assert "def run_training" in spec_code
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/executors/test_modal.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aegis.executors'"

**Step 3: Write minimal implementation**

```python
# aegis/executors/__init__.py
from aegis.executors.modal_runner import ModalExecutor

__all__ = ["ModalExecutor"]

# aegis/executors/modal_runner.py
"""
Modal-based remote executor for GPU training.
Requires: MODAL_TOKEN env var
"""

import os
import tempfile
from typing import Optional

# modal is optional - only required for remote execution
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None


class ModalExecutor:
    """
    Execute training scripts on Modal GPU infrastructure.
    """

    def __init__(self):
        if not MODAL_AVAILABLE:
            raise RuntimeError(
                "Modal not installed. Install with: pip install modal"
            )
        self.app = None

    def _generate_app(self) -> str:
        """Generate the Modal app code."""
        return '''
import modal
import tempfile
import subprocess

app = modal.App("aegis-ml-training")

@app.function(
    image=modal.Image.debian_slim()
        .pip_install("torch", "transformers", "peft", "datasets", "accelerate"),
    gpu="a10g",
    timeout=600,
)
def run_training(script_code: str, spec_json: str) -> dict:
    """Execute generated training script in Modal GPU container."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=500
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {
            "error": str(e),
            "returncode": -1
        }
'''

    def execute(
        self,
        script_code: str,
        spec_json: str,
        gpu: str = "a10g"
    ) -> dict:
        """
        Execute training script on Modal.

        Args:
            script_code: Python code to execute
            spec_json: Training spec as JSON string
            gpu: GPU type (a10g, t4, a100)

        Returns:
            Dict with stdout, stderr, returncode
        """
        if not MODAL_AVAILABLE:
            return {
                "error": "Modal not available",
                "returncode": -1
            }

        # For MVP, return a mock result
        # TODO: Implement actual Modal execution
        return {
            "stdout": "Training complete (mock)",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "train_loss": 1.2,
                "eval_loss": 1.5
            },
            "duration_sec": 300,
            "model_path": "/tmp/mock_model"
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/executors/test_modal.py -v`
Expected: PASS (all tests pass, with skip if modal not installed)

**Step 5: Commit**

```bash
git add aegis/executors/ tests/executors/
git commit -m "feat: add Modal executor skeleton for remote GPU execution"
```

---

### Task 7.2: Update Execute Node

**Files:**
- Modify: `aegis/graph.py`
- Modify: `tests/test_graph.py`

**Step 1: Write the test**

```python
# Add to tests/test_graph.py

def test_graph_execute_node_uses_modal():
    from aegis.models.state import TrainingSpec, RunEvent

    state = AegisState(
        spec=TrainingSpec(
            method="lora",
            model_name="tinyllama/tinyllama-272m",
            dataset_path="./data/sample.jsonl"
        ),
        generated_code="# Training code here"
    )

    # Import execute_node from graph
    from aegis.graph import execute_node

    new_state = execute_node(state)

    # Should have execution result
    assert new_state.execution_result is not None
```

**Step 2: Run test to verify it passes**

The test should pass with the stub implementation. Update the execute_node in graph.py to add a comment about Modal integration.

**Step 3: Update graph.py execute_node**

```python
# In aegis/graph.py, replace the execute_node function:

def execute_node(state: AegisState) -> AegisState:
    """Execute training script - local or remote."""
    from aegis.models.state import RunEvent

    if state.generated_code is None:
        event = RunEvent(
            phase="execute",
            status="failed",
            message="No code to execute"
        )
        return state.model_copy(update={"events": state.events + [event]})

    # For MVP: use mock execution
    # TODO: Integrate ModalExecutor for real GPU execution
    mock_result = {
        "metrics": {"train_loss": 1.2, "eval_loss": 1.5},
        "model_path": "/tmp/mock_model",
        "duration_sec": 300
    }

    event = RunEvent(
        phase="execute",
        status="completed",
        message="Mock execution completed (TODO: integrate Modal)",
        data=mock_result
    )

    return state.model_copy(
        update={
            "execution_result": mock_result,
            "events": state.events + [event]
        }
    )
```

**Step 4: Commit**

```bash
git add aegis/graph.py tests/test_graph.py
git commit -m "feat: add Modal integration TODO stub to execute node"
```

---

## Phase 8: Documentation & Polish

### Task 8.1: README

**Files:**
- Create: `README.md`

**Step 1: Create README**

```markdown
# 🛡️ Aegis-ML

> **Production-Grade AI Reliability Engineer** — Safe, observable fine-tuning orchestration with FinOps gates and auto-healing.

Aegis-ML is a LangGraph-based orchestration layer that brings production discipline to LLM fine-tuning. Think of it as the "adult in the room" for ML workflows — it checks costs before running, sandboxes execution, validates outputs, and generates audit reports.

## Why Aegis-ML?

Fine-tuning LLMs is easy. Doing it **safely, cheaply, and observably** is hard. Aegis-ML solves:

- 💸 **Surprise cloud bills** — Cost estimation and approval gates before GPU time
- 🔄 **Failed training runs** — Auto-healing with spec mutation and bounded retries
- 📦 **Unsafe execution** — Sandboxed training with Modal/E2B
- 📊 **No audit trail** — Complete lineage reports for compliance and review

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run the UI
streamlit run ui/streamlit_app.py
```

## Architecture

Aegis-ML is built on **LangGraph** for stateful orchestration:

```
┌─────────────────────────────────────────────────────────────────┐
│                     AEGIS GRAPH                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  START → parse_intent → estimate_cost → budget_gate            │
│                                      │                           │
│                         [approve]   [optimize]   [cancel]       │
│                               │           │           │           │
│                               ▼           ▼           END       │
│                        generate_code  remediate_spec             │
│                               │           │                       │
│                               ▼           │                       │
│                            execute ◄──────┘                       │
│                               │                                   │
│                          run_evals                               │
│                               │                                   │
│                          eval_gate                               │
│                    [pass] /     \\ [fail]                        │
│                      ▼         \\   ▼                            │
│                 write_report  remediate_spec ──┐                 │
│                      │                         │                 │
│                      ▼                         │                 │
│                     END ◄───────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `TrainingSpec` | Pydantic model defining training intent |
| `CostProfiler` | Deterministic cost/VRAM estimation |
| `BudgetGate` | HITL approval for expensive runs |
| `TemplateCodeGenerator` | Jinja2-based script generation |
| `ModalExecutor` | Remote GPU execution (sandboxed) |
| `EvalNode` | Post-training validation (loss, safety) |
| `ReportGenerator` | Markdown lineage artifacts |

## Features

### FinOps Gate 🛡️
- Automatic cost estimation before training
- Soft thresholds for auto-approval
- HITL approval for expensive runs
- Complete cost breakdown in reports

### Auto-Healing 🔄
- Intelligent spec mutation (batch size, method, LR)
- Bounded retry loops (configurable)
- Routes failures to remediation or termination

### Sandboxed Execution 📦
- Modal integration for GPU isolation
- Local Docker option for development
- E2B-ready architecture

### Observability 📊
- Append-only event log
- LangSmith tracing (optional)
- Generated REPORT.md per run

## Example Usage

```python
from aegis.graph import build_aegis_graph
from aegis.models.state import AegisState

graph = build_aegis_graph()
state = AegisState()

# Start fine-tuning
config = {"configurable": {"thread_id": "my-run"}}
result = graph.invoke(state, config)

# Check results
print(result.final_report)  # Full markdown report
```

## Configuration

Set these environment variables:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for LangSmith tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=aegis-ml

# Optional (for Modal execution)
MODAL_TOKEN_TOKEN=...
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=aegis --cov-report=html

# Format code
ruff format aegis tests
```

## Roadmap

- [ ] Full Modal integration (currently stub)
- [ ] E2B sandboxing for code eval
- [ ] LLM-based code generation (v2)
- [ ] Custom evaluation metrics
- [ ] Multi-GPU support
- [ ] Distributed training

## License

MIT

---

Built with ❤️ for production-safe ML
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with architecture and usage"
```

---

### Task 8.2: CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Create CLAUDE.md**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aegis-ML is a LangGraph-based AI Reliability Engineer that orchestrates LLM fine-tuning with safety gates. The architecture is spec-first — `TrainingSpec` Pydantic model is the source of truth, and all mutations go through immutable `model_copy()` updates.

## Key Commands

```bash
# Run tests
pytest

# Run specific test
pytest tests/path/to/test.py::test_name -v

# Run Streamlit UI
streamlit run ui/streamlit_app.py

# Format code
ruff format aegis tests
```

## Architecture Principles

1. **Immutable state**: All node functions return `state.model_copy(update={...})`, never mutate in place
2. **Spec-first**: `TrainingSpec` is the source of truth; code is generated FROM spec
3. **Bounded loops**: All remediation paths respect `retry_count < max_retries`
4. **Event logging**: Every node appends a `RunEvent` to `state.events` for auditability

## File Structure

```
aegis/
├── models/state.py      # Pydantic models: TrainingSpec, AegisState, etc.
├── nodes/               # LangGraph node functions
├── edges/               # Conditional routing functions
├── profilers/           # Cost/VRAM estimation
├── generators/          # Template-based code generation
├── executors/           # Modal/Docker execution
└── graph.py             # Main LangGraph construction
```

## Important Patterns

### Writing a Node

All nodes follow this pattern:

```python
def my_node(state: AegisState, **kwargs) -> AegisState:
    # Validate prerequisites
    if state.spec is None:
        event = RunEvent(phase="my_node", status="failed", message="...")
        return state.model_copy(update={"events": state.events + [event]})

    # Do work
    result = do_something()

    # Log success
    event = RunEvent(phase="my_node", status="completed", message="...")
    updates = {"events": state.events + [event], "result": result}
    return state.model_copy(update=updates)
```

### Writing an Edge

Edges are simple routing functions that return node names as strings:

```python
def my_routing(state: AegisState) -> str:
    if condition:
        return "next_node_a"
    else:
        return "next_node_b"  # or "__end__"
```

## Testing

- All nodes have unit tests in `tests/nodes/`
- All edges have unit tests in `tests/edges/`
- Integration tests for the graph are in `tests/test_graph.py`
- Always write tests before implementation (TDD)
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with architecture guidance"
```

---

### Task 8.3: PyProject.toml / Setup

**Files:**
- Create: `pyproject.toml`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "aegis-ml"
version = "0.1.0"
description = "Production-Grade AI Reliability Engineer for LLM fine-tuning"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "jinja2>=3.1.0",
    "modal>=0.64.0",
    "streamlit>=1.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with dependencies and tool config"
```

---

### Task 8.4: Example Data File

**Files:**
- Create: `data/sample.jsonl`

**Step 1: Create sample data**

```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Machine learning models require careful fine-tuning for optimal performance."}
{"text": "Aegis-ML provides safety gates for production AI workflows."}
{"text": "LangGraph enables stateful orchestration of AI agents."}
{"text": "Cost estimation prevents surprise cloud bills during training."}
```

**Step 2: Commit**

```bash
git add data/sample.jsonl
git commit -m "data: add sample training data for demo"
```

---

## Phase 9: Final Integration & Verification

### Task 9.1: End-to-End Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write the failing test**

```python
# tests/test_e2e.py
import pytest
from aegis.graph import build_aegis_graph
from aegis.models.state import AegisState


def test_full_graph_flow():
    """Test complete happy path from intent to report."""
    graph = build_aegis_graph()
    initial_state = AegisState()

    # This is a synchronous invoke - the full graph should complete
    # since we use auto-approve under the soft threshold
    config = {"configurable": {"thread_id": "e2e-test"}}
    result = graph.invoke(initial_state, config)

    # Verify spec was parsed
    assert result.spec is not None
    assert result.spec.method in ["lora", "qlora", "full_finetune"]

    # Verify cost was estimated
    assert result.cost_estimate is not None
    assert result.cost_estimate.estimated_cost_usd > 0

    # Verify code was generated
    assert result.generated_code is not None
    assert len(result.generated_code) > 0

    # Verify execution completed (mock)
    assert result.execution_result is not None

    # Verify evals ran
    assert result.eval_result is not None

    # Verify report was generated
    assert result.final_report is not None
    assert "# Aegis-ML Training Report" in result.final_report


def test_graph_event_log_completeness():
    """Verify all phases are logged."""
    graph = build_aegis_graph()
    result = graph.invoke(AegisState(), config={"configurable": {"thread_id": "log-test"}})

    phases = {e.phase for e in result.events}
    expected_phases = {
        "parse_intent",
        "estimate_cost",
        "budget_gate",
        "generate_code",
        "execute",
        "run_evals",
        "write_report"
    }

    assert expected_phases.issubset(phases)
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_e2e.py -v`
Expected: PASS (all tests)

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end integration tests"
```

---

### Task 9.2: Final Verification

**Files:**
- None (verification step)

**Step 1: Run all tests**

```bash
pytest -v --tb=short
```

Expected: All tests pass

**Step 2: Verify imports work**

```bash
python -c "from aegis.graph import build_aegis_graph; print('OK')"
```

Expected: Prints "OK"

**Step 3: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: final cleanup and fixes"
```

---

### Task 9.3: Tag Release

**Step 1: Create tag**

```bash
git tag -a v0.1.0 -m "Initial MVP release of Aegis-ML"
git push origin v0.1.0
```

---

## Summary

This implementation plan builds Aegis-ML in 9 phases:

1. **Foundation** — Pydantic state models and cost profiler
2. **Code Generation** — Jinja2 template-based script generator
3. **Nodes** — All LangGraph node functions
4. **Edges** — Conditional routing logic
5. **Graph** — Main LangGraph construction
6. **UI** — Streamlit interface
7. **Modal** — Remote GPU execution skeleton
8. **Docs** — README, CLAUDE.md, setup files
9. **Verification** — End-to-end testing

Each phase follows TDD: write failing test, implement, verify passing, commit.

Total: ~45 atomic commits building to a complete MVP.
