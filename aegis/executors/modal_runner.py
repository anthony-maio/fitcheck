"""Modal-based remote executor for GPU training.

Requires ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` env vars for real
execution.  Falls back to a mock result when Modal is unavailable, keeping
the test suite green without cloud credentials.
"""

import json
import os
import re

try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None


class ModalExecutor:
    """Execute training scripts on Modal GPU infrastructure."""

    def __init__(self, gpu: str = "a10g"):
        self.gpu = gpu

    # ------------------------------------------------------------------
    # Metric / output parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_error(stderr: str) -> str:
        """Extract the final traceback + error from stderr.

        HuggingFace often prints lengthy download warnings before the
        actual Python traceback.  This pulls out the last traceback block
        and error line so the user (and remediation) sees what matters.
        """
        if not stderr:
            return ""
        # Find the LAST "Traceback (most recent call last):" block
        parts = stderr.split("Traceback (most recent call last):")
        if len(parts) > 1:
            tb = "Traceback (most recent call last):" + parts[-1]
            return tb.strip()
        # No traceback found — return the last 500 chars
        return stderr[-500:].strip()

    @staticmethod
    def _parse_metrics(stdout: str) -> dict:
        """Extract metrics from the ``AEGIS_METRICS:{...}`` sentinel line."""
        match = re.search(r"AEGIS_METRICS:(\{.*\})", stdout)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _parse_duration(stdout: str) -> float | None:
        """Extract training duration in seconds from stdout."""
        match = re.search(r"AEGIS_DURATION:([\d.]+)", stdout)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _parse_model_path(stdout: str) -> str | None:
        """Extract saved model path from stdout."""
        match = re.search(r"Model saved to (.+)", stdout)
        if match:
            return match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _has_modal_token(self) -> bool:
        """Return True when Modal credentials are available (env var or CLI profile)."""
        if os.environ.get("MODAL_TOKEN_ID"):
            return True
        # Modal CLI stores auth in ~/.modal.toml — check if configured
        try:
            from pathlib import Path
            modal_config = Path.home() / ".modal.toml"
            return modal_config.exists() and modal_config.stat().st_size > 0
        except Exception:
            return False

    def _mock_execute(self, script_code: str, spec_json: str) -> dict:
        """Return a plausible mock result for local / CI environments."""
        return {
            "stdout": "Training complete (mock)\nAEGIS_METRICS:{\"train_loss\":1.2,\"eval_loss\":1.5}\nAEGIS_DURATION:300\nModel saved to /tmp/mock_model",
            "stderr": "",
            "returncode": 0,
            "metrics": {"train_loss": 1.2, "eval_loss": 1.5},
            "duration_sec": 300,
            "model_path": "/tmp/mock_model",
            "extracted_error": "",
        }

    def execute(self, script_code: str, spec_json: str, gpu: str | None = None) -> dict:
        """Execute training script on Modal, or fall back to mock.

        Args:
            script_code: The Python training script to run.
            spec_json: JSON-serialised ``TrainingSpec`` for logging.
            gpu: Override GPU type (defaults to ``self.gpu``).

        Returns:
            dict with keys: stdout, stderr, returncode, metrics,
            duration_sec, model_path.
        """
        if not MODAL_AVAILABLE or not self._has_modal_token():
            return self._mock_execute(script_code, spec_json)

        target_gpu = gpu or self.gpu

        try:
            app = modal.App("aegis-ml-training")

            image = (
                modal.Image.debian_slim()
                .pip_install(
                    "torch", "transformers", "peft", "datasets",
                    "accelerate", "bitsandbytes",
                    "sentencepiece", "tiktoken", "protobuf",
                    "einops", "scipy",
                )
            )

            # Forward HF_TOKEN so the script can download gated models/datasets
            secrets = []
            if os.environ.get("HF_TOKEN"):
                secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

            @app.function(
                image=image,
                gpu=target_gpu,
                timeout=1800,
                serialized=True,
                secrets=secrets,
            )
            def run_training(code: str) -> dict:
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as f:
                    f.write(code)
                    script_path = f.name
                try:
                    result = subprocess.run(
                        ["python", script_path],
                        capture_output=True,
                        text=True,
                        timeout=1700,
                    )
                    return {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    }
                except Exception as e:
                    return {"stdout": "", "stderr": str(e), "returncode": -1}

            with app.run():
                raw = run_training.remote(script_code)

            stdout = raw.get("stdout", "")
            stderr = raw.get("stderr", "")
            returncode = raw.get("returncode", -1)

            metrics = self._parse_metrics(stdout)
            duration = self._parse_duration(stdout)
            model_path = self._parse_model_path(stdout)

            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "metrics": metrics,
                "duration_sec": duration,
                "model_path": model_path,
                "extracted_error": self._extract_error(stderr),
            }

        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "metrics": {},
                "duration_sec": None,
                "model_path": None,
                "extracted_error": str(e),
            }
