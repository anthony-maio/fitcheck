# Aegis: Fine-Tuning Pre-Flight Engine

> Design document for the Aegis reboot from LangGraph orchestration demo to
> adoptable open-source tool.  Derived from collaborative brainstorming session
> on 2026-02-13.

---

## 1. Problem Statement

The feedback loop between choosing a fine-tuning configuration and knowing
whether it will work is unacceptably slow.  Engineers guess at configs, wait
10-30 minutes to hit OOM or divergence, tweak, and repeat.  On cloud GPUs this
wastes money.  On local hardware this wastes hours.

The information needed to predict outcomes already exists: model architecture
metadata, GPU memory specifications, known training dynamics.  Nobody has
assembled it into a single tool that gives a clear, architecture-aware
prediction with explicit confidence bounds.

### Real-world example

Training Eve-2-MoE-272M on an H100 SXM (140 GB):  the session involved
guessing at batch size 32 vs 4, debating gradient accumulation steps, choosing
between fp16 and bf16, and wondering whether the full 1.4M-row dataset would
fit in a single epoch.  None of those questions required running the training
job -- they're all computable from the model architecture, the dataset, and the
hardware spec.  A tool that answered them in seconds would have saved an hour of
setup and debugging.

---

## 2. Product Definition

**Aegis is a fine-tuning pre-flight engine.**  Given a model, a training
method, a dataset, and target hardware, it computes a detailed, architecture-
aware prediction of VRAM usage, optimal batch configuration, estimated cost and
duration, the most likely failure modes, and pre-computed fallback
configurations.

When you also let it run the training, it uses that same pre-computed knowledge
to auto-recover from failures without guessing.

**What it is not.**  Aegis is not a training framework.  It does not replace
Axolotl, Unsloth, or HuggingFace Trainer.  It sits *in front of* them --
telling you what to configure and whether it's worth running.  Think of it as
the flight plan, not the autopilot.

**Positioning.**  The fine-tuning ecosystem has a gap.  HuggingFace gives you
model configs.  Unsloth/Axolotl give you training wrappers.  Cloud providers
give you GPU specs.  Nobody connects those three things into "here's exactly
what will happen when you combine *this model* with *this hardware* with *this
method*."  People do that math in their heads, in spreadsheets, or by trial and
error.

### Adoption vector

The CLI report output is the product.  When someone pastes the `aegis plan`
output in a Discord channel, people ask "what tool is this?"  That's the
adoption path.

---

## 3. v1 Scope

### Architectures (computed from first principles)

| Family | Models covered | Notes |
|--------|---------------|-------|
| LLaMA-family | Llama 3.x, Qwen 2.5, Mistral, Phi | Same decoder-only arch, minor GQA variations |
| Gemma | Gemma 2, Gemma 3 | Slightly different attention setup |
| MoE | Mixtral 8x7B, Qwen-MoE | **High-value differentiator** -- VRAM math is most confusing here.  47B total params with 13B active is where everyone's intuition fails |

### Training methods

- Full fine-tune (fp16/bf16/fp32)
- LoRA (configurable rank, targets)
- QLoRA (NF4 base + bf16 adapters)

### Hardware

- Single GPU: RTX 3090, RTX 4090, A10G, A100 40GB, A100 80GB, H100 80GB
- Multi-GPU deferred to v1.1

### Dataset awareness

The solver accepts a dataset path or HuggingFace dataset reference, samples it,
tokenizes with the model's tokenizer (including chat template overhead), and
computes sequence length statistics (min, mean, p50, p95, p99, max).  p95 is
used by default for VRAM estimation.

The analyzer detects common formats (Alpaca, ShareGPT, raw text) and applies
the appropriate tokenizer with the model's chat template.  Without this, p95
estimates can be off by 30-40% for chat-format data due to template overhead.

### Delivery

- **Python library** -- `from aegis import plan`
- **CLI** -- `aegis plan`, `aegis run`, `aegis benchmark`
- **LangGraph orchestrator** -- opt-in execution layer on top of the planner

### v1 cut list

- No encoder-decoder (T5, BART)
- No vision models (LLaVA, Qwen-VL)
- No embedding model training
- No multi-node distributed training
- No custom kernel estimation

---

## 4. Solver Architecture

The solver is a computation pipeline with six stages.  Each stage is
independently testable with clearly defined inputs and outputs.

### Stage 1: Model Resolver

Takes a HuggingFace model identifier (e.g. `meta-llama/Llama-3.1-8B`).  Pulls
`config.json` via `huggingface_hub` (with built-in caching for offline-after-
first-fetch).  Maps the `architectures` field to an architecture family class.

Returns a normalized `ModelProfile`: parameter count, hidden size, num layers,
num attention heads, num KV heads (for GQA), intermediate size, vocab size, and
-- for MoE -- num experts, experts per token, and router config.

### Stage 2: Dataset Analyzer

Takes a local path or HuggingFace dataset reference.  Samples N rows (default
1000, configurable).  Tokenizes using the model's tokenizer with the
appropriate chat template.  Detects common formats (Alpaca, ShareGPT, raw text).

Returns a `DatasetProfile`: sequence length distribution (min, mean, p50, p95,
p99, max), row count, and detected format.

### Stage 3: Hardware Registry

A curated registry of GPU specifications.  Each entry contains: total VRAM,
memory bandwidth, FP16/BF16 TFLOPS, and a **measured overhead margin**
(benchmarked CUDA context + PyTorch framework reservation).

```
RTX 3090:  24.0 GB total -> 22.8 GB usable (1.2 GB overhead)
RTX 4090:  24.0 GB total -> 22.5 GB usable (1.5 GB overhead)
A100 80GB: 80.0 GB total -> 78.5 GB usable (1.5 GB overhead)
H100 80GB: 80.0 GB total -> 78.5 GB usable (1.5 GB overhead)
```

The report shows this breakdown explicitly.  Showing the work builds trust.

Includes cloud GPU pricing for counterfactual cost display (Modal, RunPod,
Lambda Labs).

Supports optional per-user calibration via `aegis benchmark` (runs 10 training
steps, measures actual throughput).  First run is estimated; subsequent runs on
the same hardware use measured data.  This is a retention loop -- the more you
use the tool, the more accurate it gets.

### Stage 4: VRAM Computation Engine

The core of the product.  VRAM during training decomposes into six components,
each computed separately and summed.

**Component 1: Model weights.**
Depends on method and dtype:
- Full fine-tune (bf16): `total_params * 2 bytes`
- LoRA (bf16 base): `total_params * 2 bytes` + adapter weights
- QLoRA (NF4 base): `total_params * 0.55 bytes` (4-bit + quantization metadata) + adapters in bf16

For MoE: `total_params` means ALL experts, not active-per-forward-pass.
Mixtral 8x7B stores ~47B parameters regardless of top-k routing.  The solver
makes this explicit in the report.

LoRA adapter size: `rank * (in_features + out_features) * 2 bytes *
num_target_modules * num_layers`.  Defaults to targeting all attention + MLP
projections (q, k, v, o, gate, up, down) and shows how VRAM changes with fewer
targets.

> **Architectural note:** Weight memory is architecture-independent for dense
> models.  For MoE, the `MoEFamily` class must be able to override this
> calculation to support CPU offloading of non-active experts in v1.1
> (DeepSpeed, FSDP).  Structure `components.py` so that family classes can
> provide a custom `weight_memory()` override.

**Component 2: Optimizer states.**
Only allocated for *trainable* parameters:
- AdamW: 2 fp32 states per param -> `trainable_params * 8 bytes`
- 8-bit Adam (paged_adamw_8bit): `trainable_params * 2 bytes` (significant savings)

For full fine-tune: optimizer states dominate.  For LoRA: they're small because
only adapter params are trainable.  The report explains this tradeoff.

**Component 3: Gradients.**
One gradient per trainable parameter in training dtype:
- Full fine-tune bf16: `total_params * 2 bytes`
- LoRA/QLoRA: `lora_trainable_params * 2 bytes`

**Component 4: Activations.**
Where sequence length and batch size hit.  Per-layer activation memory:

```
activation_per_layer ~ batch_size * seq_len * hidden_size * factor
```

`factor` is architecture-family-specific (attention pattern, MLP width, fused
kernels).  For MoE, only top-k expert MLPs contribute per token -- activation
cost is closer to a 13B dense model than a 47B one.

Gradient checkpointing (enabled by default near the VRAM ceiling): trades
compute for memory by recomputing activations during backward pass.  Reduces
from `O(num_layers)` to roughly `O(sqrt(num_layers))`.

At high LoRA ranks (128+), adapter activations during the forward pass become
non-negligible.  The formula includes a rank-proportional overhead term.

MoE router computation adds `batch_size * seq_len * num_experts` for gating
logits plus token-routing indices.  Small relative to expert activations but
included for completeness.

**Component 5: Logits buffer (loss computation).**
The logits tensor for cross-entropy is `batch_size * seq_len * vocab_size` in
float32.  For a 128k vocabulary model: `1 * 4096 * 128000 * 4 bytes = ~2 GB`
per sample.  At batch size 4 that's 8 GB just for the loss computation.

This is one of the most common surprise OOMs.  Some frameworks use chunked
cross-entropy (Unsloth does, HF Trainer does not by default).  The solver
computes this explicitly and flags when it's the bottleneck with framework-
specific mitigation advice.

**Component 6: Eval KV-cache spike.**
Periodic evaluation steps allocate KV-cache for the full sequence length.  For
long-context fine-tuning this spikes above training steady-state.  The solver
flags: "Training fits at 21.3 GB, but eval steps at seq_len=4096 will spike to
23.8 GB.  Consider `eval_accumulation_steps` or shorter eval sequences."

This prevents the failure mode where training runs fine for 500 steps and then
OOMs on the first eval checkpoint.

**The sum:**

```
total = weights + optimizer_states + gradients + activations
        + logits_buffer + eval_kv_spike (periodic)
```

**Confidence bounds** use two separate margins:
- **Systematic margin:** framework overhead, CUDA context, allocator behavior.
  Roughly constant per GPU class, calibrated once.  Narrows the usable VRAM
  estimate.
- **Dynamic margin:** memory fragmentation, peak transient allocations during
  optimizer steps, cuDNN workspace buffers.  Varies during the run.  Creates
  the range in the final prediction.

Separating these enables better advice: "You have 1.8 GB of headroom after
systematic overhead.  Dynamic peaks may consume up to 0.9 GB.  This config will
probably fit but has no safety margin -- reduce batch size by 1 for a reliable
run."

### Stage 5: Config Optimizer

A constraint solver.  Given the VRAM budget, it finds the best training
configuration.

Decision variables:
- Micro batch size (note: this is what determines activation memory, not
  effective batch size -- gradient accumulation doesn't multiply activations)
- Gradient accumulation steps
- Gradient checkpointing on/off
- LoRA rank and target module selection
- Chunked cross-entropy (framework-dependent)
- Eval strategy (accumulation steps, shorter eval seqs)

The solver produces three tiers:

1. **Recommended** -- fits with comfortable margin, optimized for throughput.
2. **Aggressive** -- maximum throughput, fits but with <1 GB headroom.
   Explicitly labeled: "expect occasional OOM on long sequences.  Appropriate
   for local hardware where an OOM means a restart, not a cloud bill."
3. **Fallback chain** -- pre-computed "if OOM, try this next" sequence, each
   entry proven to fit.  Ordered by expected training quality.

The solver uses a staged approach rather than full combinatorial search:
1. Fix method and LoRA rank (user choice or sensible default)
2. Binary search on batch size for the VRAM budget
3. Compute gradient accumulation to hit target effective batch size
4. Check if relaxing rank opens better configs
5. Build fallback chain from the sorted list of configs that fit

The fallback chain is the key differentiator.  When the orchestrator catches an
OOM, it pulls the next pre-computed config rather than guessing.

### Stage 6: Report Generator

Produces the CLI output and a structured `PlanReport` Pydantic model.

The report includes:
- **Model summary:** architecture, parameter count, vocab size
- **Dataset summary:** row count, sequence length distribution, which percentile is used and why ("p95 chosen over max because padding to max wastes 73% of tokens")
- **Hardware summary:** total VRAM, overhead breakdown, usable VRAM
- **"What you're training" sanity check:** trainable param % (e.g., "fine-tuning 0.12% of parameters -- 9.4M of 8.03B"), samples seen per epoch, overfit risk flag if dataset is too small relative to model size
- **VRAM breakdown:** each component itemized, total with confidence range, headroom assessment
- **Recommended config** with reasoning annotations everywhere (e.g., "Grad checkpointing: OFF -- enough headroom", "Using p95 (1,024 tokens) -- set max_seq_length=1024 and truncate outliers, or use packing")
- **Estimates:** throughput range (not false precision -- "~1,500-2,200 tok/sec, varies with framework"), time, cost with cloud counterfactual ("$0.00 local -- equivalent cloud run: ~$0.85 on A10G, ~$2.40 on A100")
- **Risk assessment:** VRAM margin, eval spike warning, training dynamics sanity
- **Fallback configs** with VRAM for each tier

Throughput estimates carry an explicit asterisk in v1.  For precise throughput,
`aegis benchmark` runs 10 calibration steps and stores measured tok/sec per
hardware + method combination.

---

## 5. CLI Interface

```
aegis plan "qlora llama-3.1-8b on 3090 with ./data/support.jsonl"
aegis plan --model meta-llama/Llama-3.1-8B --method qlora --gpu 3090 --dataset ./data/support.jsonl
aegis benchmark --gpu 3090 --method qlora --model meta-llama/Llama-3.1-8B
aegis run [options]   # execute with auto-recovery from fallback chain
```

Natural language parsing in `aegis plan "..."` extracts model ID, method,
hardware, and dataset path.  Structured flags are the fallback for precision.

---

## 6. Example Report Output

```
$ aegis plan "qlora llama-3.1-8b on 3090 with ./data/support.jsonl"

Model: meta-llama/Llama-3.1-8B
  Architecture: LlamaForCausalLM (dense decoder)
  Parameters: 8.03B | Vocab: 128,256 | Layers: 32

Dataset: ./data/support.jsonl (10,247 rows)
  Sequence lengths: mean=342  p95=1,024  max=3,891
  Format: ShareGPT (detected) -- tokenized with Llama chat template
  Using p95 (1,024 tokens) -- p95 chosen over max (3,891) because
  padding to max would waste 73% of tokens. Set max_seq_length=1024
  and truncate outliers, or use packing.

Hardware: NVIDIA RTX 3090
  Total: 24.0 GB | Framework overhead: 1.2 GB | Usable: 22.8 GB

What you're training:
  Method: QLoRA (NF4 base + bf16 adapters)
  Trainable: 9.4M / 8.03B parameters (0.12%)
  LoRA rank 16, targets: q,k,v,o,gate,up,down (all attn + MLP)
  At effective batch 16 with 10,247 rows: ~4.8 samples/epoch
  3 epochs = each sample seen ~14.4 times -- reasonable for 10k rows

VRAM Breakdown (QLoRA rank 16, bs=4, seq=1024):
  Base model (NF4)            4.42 GB
  LoRA adapters (bf16)        0.08 GB
  Optimizer states (AdamW)    0.64 GB
  Gradients                   0.08 GB
  Activations (bs=4)          5.21 GB
  Logits buffer (128k vocab)  2.00 GB
  ---
  Total                      12.43 GB
  Dynamic margin             +/- 0.8 GB
  Predicted range        11.6 - 13.2 GB  (58% of 3090)
  Headroom: 9.6 GB -- comfortable

Recommended Config:
  Micro batch: 4 | Grad accum: 4 | Effective: 16
  Seq len: 1,024 (your data p95)
  Grad checkpointing: OFF (9.6 GB headroom -- not needed)
  Optimizer: AdamW (paged_adamw_8bit saves 0.3 GB if needed)

Aggressive Config (max throughput, <1 GB headroom):
  Micro batch: 8 | Grad accum: 2 | Effective: 16
  17.6 GB predicted -- fits but OOM likely on long sequences
  Appropriate for local hardware where restart is cheap

Estimates:
  Throughput: ~1,500 - 2,200 tok/sec (estimated, run `aegis benchmark` to calibrate)
  Time: 3 epochs ~ 35-50 min
  Cost: $0.00 (local) -- equivalent cloud: ~$0.85 A10G, ~$2.40 A100

Risks:
  [ok] VRAM: 9.6 GB headroom -- no OOM risk
  [!!] Eval spike: eval at max seq (3,891) -> ~18.1 GB
       Set eval_accumulation_steps=4 or max_eval_samples with shorter seqs
  [ok] Training dynamics: effective batch 16 is reasonable for 10k rows

Fallbacks (pre-computed, if primary OOMs):
  1. bs=2, grad_accum=8 -> 9.8 GB  (same effective batch, same quality)
  2. Enable grad checkpointing -> 7.1 GB  (~15% slower throughput)
  3. Rank 8 + grad ckpt + bs=1 -> 5.9 GB  (fits RTX 3060 12 GB)
```

---

## 7. Module Structure

```
aegis/
|-- models/
|   |-- state.py              # AegisState (updated with new fields)
|   |-- profiles.py           # ModelProfile, DatasetProfile, HardwareSpec
|   +-- results.py            # VRAMBreakdown, SolverResult, PlanReport
|
|-- profilers/
|   |-- vram/
|   |   |-- engine.py         # VRAMEstimator -- top-level orchestrator
|   |   |-- components.py     # Shared calculators: weights, optimizer, gradients
|   |   +-- families/
|   |       |-- base.py       # ArchitectureFamily protocol
|   |       |-- llama.py      # LlamaFamily (Llama, Qwen, Mistral, Phi)
|   |       |-- gemma.py      # GemmaFamily
|   |       +-- moe.py        # MoEFamily (Mixtral, Qwen-MoE)
|   |-- solver.py             # ConfigSolver -- constraint optimization
|   +-- sanity.py             # TrainingSanityCheck -- overfit, dynamics flags
|
|-- hardware/
|   |-- registry.py           # Static GPU specs + overhead margins
|   |-- pricing.py            # Cloud GPU pricing (Modal, RunPod, Lambda)
|   +-- calibration.py        # aegis benchmark result store
|
|-- datasets/
|   +-- analyzer.py           # Sample, tokenize, compute seq_len stats
|
|-- hub/
|   +-- resolver.py           # HF config.json -> ModelProfile
|
|-- report/
|   +-- formatter.py          # CLI report renderer
|
|-- cli.py                    # aegis plan / run / benchmark
|
|-- nodes/                    # LangGraph nodes (slimmed, consume solver output)
|-- edges/                    # LangGraph edges (simplified)
|-- graph.py                  # Graph wiring (remediation uses fallback chain)
|-- generators/               # Jinja2 templates (config from solver)
+-- executors/                # Local runner (v1), Modal stub (v1.1)
```

### Key interfaces

```python
# aegis/profilers/vram/families/base.py
class ArchitectureFamily(Protocol):
    """Each family knows how to compute VRAM for its architecture."""

    def activation_memory(
        self, config: ModelProfile, batch_size: int,
        seq_len: int, grad_checkpointing: bool,
    ) -> ComponentEstimate: ...

    def logits_buffer(
        self, config: ModelProfile, batch_size: int, seq_len: int,
    ) -> ComponentEstimate: ...

    def kv_cache_eval(
        self, config: ModelProfile, eval_batch_size: int, eval_seq_len: int,
    ) -> ComponentEstimate: ...

    # Optional override for families (MoE) that need custom weight calculation
    def weight_memory(
        self, config: ModelProfile, method: Method, dtype: DType,
    ) -> ComponentEstimate | None:
        """Return None to use the default shared calculation."""
        return None
```

```python
# aegis/profilers/vram/components.py
def weight_memory(
    total_params: int, method: Method, dtype: DType,
    lora_rank: int = 0, lora_targets: int = 0,
) -> ComponentEstimate: ...

def optimizer_memory(
    trainable_params: int, optimizer: str = "adamw",
) -> ComponentEstimate: ...

def gradient_memory(
    trainable_params: int, dtype: DType,
) -> ComponentEstimate: ...
```

```python
# aegis/profilers/vram/engine.py
class VRAMEstimator:
    """Orchestrates all components into a full VRAM breakdown."""

    def estimate(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        dataset: DatasetProfile,
        method: Method,
        batch_size: int,
        lora_config: LoRAConfig | None = None,
    ) -> VRAMBreakdown: ...
```

```python
# aegis/profilers/solver.py
class ConfigSolver:
    """Finds optimal config given constraints.  Produces three tiers."""

    def solve(
        self,
        model: ModelProfile,
        hardware: HardwareSpec,
        dataset: DatasetProfile,
        method: Method,
    ) -> SolverResult:
        # 1. Fix method + LoRA rank
        # 2. Binary search on batch size
        # 3. Compute grad_accum for target effective batch
        # 4. Check if relaxing rank opens better configs
        # 5. Build fallback chain
        # 6. Run sanity checks
        ...
```

---

## 8. Codebase Migration

### What stays
- `AegisState` and the Pydantic model pattern -- solid foundation, needs new fields
- The graph topology concept -- parse -> plan -> gate -> execute -> eval -> report
- The test structure and TDD discipline
- Jinja2 template approach for code generation (templates consume solver output)

### What gets rewritten
- `aegis/profilers/cost.py` -- becomes the entire VRAM computation engine
- `aegis/nodes/intent.py` -- parses model IDs, hardware specs, dataset paths
- `aegis/nodes/remediate.py` -- pops next config from pre-computed fallback chain

### What's new
- `aegis/profilers/vram/` -- architecture family registry, component calculators, engine
- `aegis/hardware/` -- GPU registry, pricing, calibration
- `aegis/datasets/` -- dataset sampling and analysis
- `aegis/hub/` -- HuggingFace Hub integration
- `aegis/report/` -- CLI report formatter
- `aegis/cli.py` -- Typer/Click CLI
- `aegis/profilers/solver.py` -- config optimization
- `aegis/profilers/sanity.py` -- training dynamics sanity checker

### What gets demoted
- Streamlit UI -- still useful but not the adoption path.  CLI first, web later
- Modal executor -- keep stub, v1.1 concern.  Local execution is the v1 story
- Budget gate HITL flow -- less relevant when primary value is pre-flight planning

### Honest assessment
~30% of current code survives structurally.  Models, test patterns, and graph
wiring carry forward.  The profiler and remediation paths are rewrites.  The
current code proved out the architecture.  Now we fill it with substance.

---

## 9. Testing Strategy

### Unit tests (run on every commit)
- Component calculators: known model configs produce expected VRAM numbers
- Architecture families: activation memory formulas verified against hand calculations
- Config solver: given fixed inputs, produces expected batch size and fallback chain
- Hardware registry: lookup returns correct specs
- Dataset analyzer: mock tokenizer produces expected seq_len distribution
- Report formatter: structured output matches expected format

### Accuracy tests (CI nightly, require network + transformers)
```python
@pytest.mark.parametrize("model_id", [
    "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b",
    "mistralai/Mixtral-8x7B-v0.1",
])
def test_parameter_count_matches_reality(model_id):
    """Load on meta device, count params, compare to our formula."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="meta"
    )
    actual = sum(p.numel() for p in model.parameters())
    predicted = VRAMEstimator().resolve_profile(model_id).total_params
    assert abs(predicted - actual) / actual < 0.01  # within 1%
```

### Integration tests
- End-to-end `aegis plan` CLI invocation produces valid report
- Solver fallback chain entries all fit within stated VRAM budgets
- Report reasoning annotations are present and non-empty

---

## 10. Implementation Sequence

### Week 1: Profiler core (proves the concept)
- `ModelProfile` from HF Hub resolver
- `components.py` shared calculators (weights, optimizer, gradients)
- `LlamaFamily` (one architecture family)
- `VRAMEstimator` producing `VRAMBreakdown` for a single config
- Meta-device accuracy test for Llama-3.1-8B
- **Exit criterion:** `estimator.estimate()` returns numbers within 5% of
  reality for Llama + QLoRA on a 3090

### Week 2: Solver and CLI (shippable milestone)
- `ConfigSolver` with batch-size sweep and fallback chain
- `HardwareSpec` registry with measured overheads
- `DatasetAnalyzer` with format detection and seq_len stats
- `aegis plan` CLI command with report formatter
- **Exit criterion:** `aegis plan "qlora llama-3.1-8b on 3090"` produces the
  full report output shown in this document

### Week 3: Breadth and polish
- `GemmaFamily` and `MoEFamily` implementations
- Training sanity checker (overfit detection, dynamics flags)
- Cloud pricing counterfactual display
- Reasoning annotations throughout the report
- Meta-device accuracy tests for Gemma and Mixtral
- **Exit criterion:** MoE report correctly shows total vs active parameters and
  produces accurate VRAM predictions

### Week 4: Integration
- Rewire LangGraph nodes to consume solver output
- `remediate_spec` uses pre-computed fallback chain
- Wire `aegis run` to local executor
- `aegis benchmark` calibration step
- Update Streamlit UI to display new report format
- **Exit criterion:** full `aegis plan` -> `aegis run` workflow with auto-
  recovery from a simulated OOM

---

## 11. Open Questions

### Package name
`aegis` is taken on PyPI (v1.1.1).  `aegis-ml` is available.  The CLI command
can still be `aegis` regardless of package name.  Decision: use `aegis-ml` on
PyPI, `aegis` as the CLI entry point.  Revisit if there's a naming conflict.

### Multi-dimensional solver
The v1 solver fixes method + LoRA rank and sweeps batch size.  A richer solver
would trade off across rank, targets, and batch size simultaneously ("rank 32
with bs=2 gives better quality than rank 16 with bs=4").  Defer full
combinatorial search to v1.1 but structure the solver interface to support it.

### CPU offloading as a fallback
For MoE models, offloading non-active expert weights to CPU is a legitimate
middle ground between "fits" and "doesn't fit."  The `MoEFamily` weight_memory
override supports this in v1.1.  For v1, all weights are assumed to be on GPU.

### Throughput estimation
VRAM is computable from first principles.  Throughput is not -- it depends on
memory bandwidth, kernel implementations, flash attention, quantization backend,
data loading.  v1 provides a range estimate.  `aegis benchmark` provides
calibrated measurements.  Do not fake precision.

---

## 12. Success Criteria

The tool is successful when:

1. A user can run `aegis plan` and get a VRAM prediction within 10% of actual
   measured usage for any supported model + method + hardware combination
2. The report output is clear enough that someone unfamiliar with the tool
   understands every number and recommendation
3. The MoE case produces predictions that are meaningfully more accurate than
   naive parameter-count-based estimation
4. The fallback chain recommendations actually work -- if the primary config
   OOMs, the first fallback fits
5. Someone screenshots the report and shares it in a Discord channel
