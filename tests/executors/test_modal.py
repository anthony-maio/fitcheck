import pytest

from aegis.executors.modal_runner import ModalExecutor


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_modal_executor_can_be_created():
    executor = ModalExecutor()
    assert executor is not None


def test_modal_executor_default_gpu():
    executor = ModalExecutor()
    assert executor.gpu == "a10g"


def test_modal_executor_custom_gpu():
    executor = ModalExecutor(gpu="a100")
    assert executor.gpu == "a100"


# ---------------------------------------------------------------------------
# Mock fallback (no Modal token)
# ---------------------------------------------------------------------------

def test_modal_executor_execute_returns_result():
    executor = ModalExecutor()
    result = executor.execute("print('hello')", "{}")
    assert result["returncode"] == 0
    assert "metrics" in result


def test_mock_fallback_without_token(monkeypatch):
    """Without MODAL_TOKEN_ID the executor should return mock results."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    executor = ModalExecutor()
    result = executor.execute("print('hello')", "{}")
    assert result["returncode"] == 0
    assert result["metrics"]["train_loss"] == 1.2
    assert result["model_path"] == "/tmp/mock_model"
    assert result["duration_sec"] == 300


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def test_parse_metrics_valid():
    stdout = 'Some output\nAEGIS_METRICS:{"train_loss":0.5,"eval_loss":0.7}\nDone'
    metrics = ModalExecutor._parse_metrics(stdout)
    assert metrics == {"train_loss": 0.5, "eval_loss": 0.7}


def test_parse_metrics_missing():
    assert ModalExecutor._parse_metrics("no sentinel here") == {}


def test_parse_metrics_invalid_json():
    assert ModalExecutor._parse_metrics("AEGIS_METRICS:{bad json}") == {}


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

def test_parse_duration_valid():
    assert ModalExecutor._parse_duration("AEGIS_DURATION:123.4") == 123.4


def test_parse_duration_missing():
    assert ModalExecutor._parse_duration("no duration") is None


# ---------------------------------------------------------------------------
# Model path parsing
# ---------------------------------------------------------------------------

def test_parse_model_path_valid():
    stdout = "Model saved to /output/llama-lora\nDone"
    assert ModalExecutor._parse_model_path(stdout) == "/output/llama-lora"


def test_parse_model_path_missing():
    assert ModalExecutor._parse_model_path("nothing here") is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_mock_result_has_required_keys():
    executor = ModalExecutor()
    result = executor._mock_execute("code", "{}")
    required_keys = {"stdout", "stderr", "returncode", "metrics", "duration_sec", "model_path", "extracted_error"}
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------

def test_extract_error_with_traceback():
    stderr = (
        "WARNING: downloading files...\n"
        "Traceback (most recent call last):\n"
        '  File "train.py", line 10, in main\n'
        "ValueError: sentencepiece not installed\n"
    )
    result = ModalExecutor._extract_error(stderr)
    assert "Traceback" in result
    assert "sentencepiece" in result
    assert "WARNING" not in result


def test_extract_error_multiple_tracebacks_takes_last():
    stderr = (
        "Traceback (most recent call last):\n"
        "  first error\n"
        "RuntimeError: first\n"
        "\n"
        "Traceback (most recent call last):\n"
        "  second error\n"
        "ValueError: second\n"
    )
    result = ModalExecutor._extract_error(stderr)
    assert "second" in result
    assert "first" not in result


def test_extract_error_no_traceback():
    result = ModalExecutor._extract_error("some random output with no traceback")
    assert result == "some random output with no traceback"


def test_extract_error_empty():
    assert ModalExecutor._extract_error("") == ""
