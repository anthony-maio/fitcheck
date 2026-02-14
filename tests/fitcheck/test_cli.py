"""CLI smoke tests using Typer's CliRunner.

These test the CLI wiring — that flags are parsed, errors are reported,
and output contains the expected sections. They do NOT test VRAM math
(that's test_solver.py and test_estimator.py).

Tests that require HF Hub are marked with @pytest.mark.network and
skipped in CI. The remaining tests use mock model profiles.
"""

from unittest.mock import patch
from typer.testing import CliRunner

from fitcheck.cli import app
from fitcheck.models.profiles import ModelProfile

runner = CliRunner()


def _mock_llama_8b() -> ModelProfile:
    return ModelProfile(
        model_id="meta-llama/Llama-3.1-8B",
        architecture="LlamaForCausalLM",
        family="llama",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        total_params=8_030_000_000,
        total_params_b=8.03,
    )


class TestCLIParsing:
    """Test that CLI flags are parsed correctly."""

    def test_missing_model_flag(self):
        result = runner.invoke(app, ["plan", "--method", "qlora", "--gpu", "3090"])
        assert result.exit_code != 0

    def test_missing_method_flag(self):
        result = runner.invoke(app, ["plan", "--model", "x", "--gpu", "3090"])
        assert result.exit_code != 0

    def test_missing_gpu_flag(self):
        result = runner.invoke(app, ["plan", "--model", "x", "--method", "qlora"])
        assert result.exit_code != 0

    def test_unknown_method(self):
        with patch("fitcheck.cli.resolve_model", return_value=_mock_llama_8b()):
            result = runner.invoke(
                app, ["plan", "--model", "test", "--method", "banana", "--gpu", "3090"]
            )
        assert result.exit_code != 0
        assert "unknown method" in result.output.lower()

    def test_unknown_gpu(self):
        with patch("fitcheck.cli.resolve_model", return_value=_mock_llama_8b()):
            result = runner.invoke(
                app, ["plan", "--model", "test", "--method", "qlora", "--gpu", "potato"]
            )
        assert result.exit_code != 0


class TestCLIOutput:
    """Test that CLI produces expected output sections."""

    @patch("fitcheck.cli.resolve_model")
    def test_qlora_plan_exits_zero(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"

    @patch("fitcheck.cli.resolve_model")
    def test_output_contains_model_section(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "Model" in result.output
        assert "8.03" in result.output

    @patch("fitcheck.cli.resolve_model")
    def test_output_contains_hardware_section(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "Hardware" in result.output
        assert "24.0" in result.output

    @patch("fitcheck.cli.resolve_model")
    def test_output_contains_vram_breakdown(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "VRAM Breakdown" in result.output
        assert "Model weights" in result.output
        assert "Optimizer states" in result.output

    @patch("fitcheck.cli.resolve_model")
    def test_output_contains_recommended_config(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "qlora", "--gpu", "3090"]
        )
        assert "Recommended Config" in result.output
        assert "Micro batch" in result.output

    @patch("fitcheck.cli.resolve_model")
    def test_full_ft_shows_does_not_fit(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app, ["plan", "--model", "test/model", "--method", "full", "--gpu", "3090"]
        )
        assert result.exit_code == 0
        assert "DOES NOT FIT" in result.output or "does not fit" in result.output.lower()

    @patch("fitcheck.cli.resolve_model")
    def test_custom_seq_len(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app,
            [
                "plan",
                "--model",
                "test/model",
                "--method",
                "qlora",
                "--gpu",
                "3090",
                "--seq-len",
                "1024",
            ],
        )
        assert result.exit_code == 0
        assert "1024" in result.output

    @patch("fitcheck.cli.resolve_model")
    def test_custom_lora_rank(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app,
            [
                "plan",
                "--model",
                "test/model",
                "--method",
                "qlora",
                "--gpu",
                "3090",
                "--lora-rank",
                "64",
            ],
        )
        assert result.exit_code == 0
        assert "64" in result.output

    @patch("fitcheck.cli.resolve_model")
    def test_fixed_batch_size(self, mock_resolve):
        mock_resolve.return_value = _mock_llama_8b()
        result = runner.invoke(
            app,
            [
                "plan",
                "--model",
                "test/model",
                "--method",
                "qlora",
                "--gpu",
                "3090",
                "--batch-size",
                "2",
            ],
        )
        assert result.exit_code == 0
