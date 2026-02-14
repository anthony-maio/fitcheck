"""Behavioral tests for the dataset analyzer.

Tests that analyze_local correctly detects dataset formats, estimates token
counts via the chars/4 heuristic, and handles edge cases in file loading.
All tests use real temporary fixture files written to disk -- no mocks.
"""

import json

import pytest

from fitcheck.datasets.analyzer import analyze_local


# ---------------------------------------------------------------------------
# Helpers for writing fixture files
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows):
    """Write a list of dicts as a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(path, data):
    """Write arbitrary data as a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """analyze_local detects dataset format from column names in the first row."""

    def test_alpaca_format_detected_from_instruction_and_output(self, tmp_path):
        """JSONL with instruction/input/output columns is detected as alpaca."""
        rows = [
            {"instruction": "Summarize this.", "input": "Some long text.", "output": "A summary."},
            {"instruction": "Translate.", "input": "Hola.", "output": "Hello."},
        ]
        path = tmp_path / "alpaca.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.detected_format == "alpaca"

    def test_alpaca_without_input_field_still_detected(self, tmp_path):
        """Alpaca detection only requires instruction + output, not input."""
        rows = [
            {"instruction": "Write a poem.", "output": "Roses are red..."},
        ]
        path = tmp_path / "alpaca_no_input.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.detected_format == "alpaca"

    def test_sharegpt_format_detected_from_conversations_list(self, tmp_path):
        """JSONL with a conversations column containing list of dicts is sharegpt."""
        rows = [
            {
                "conversations": [
                    {"from": "human", "value": "What is Python?"},
                    {"from": "gpt", "value": "Python is a programming language."},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Tell me a joke."},
                    {"from": "gpt", "value": "Why did the chicken cross the road?"},
                ]
            },
        ]
        path = tmp_path / "sharegpt.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.detected_format == "sharegpt"

    def test_raw_text_format_detected_from_text_column(self, tmp_path):
        """JSONL with just a text column is detected as raw_text."""
        rows = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Pack my box with five dozen liquor jugs."},
        ]
        path = tmp_path / "raw_text.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.detected_format == "raw_text"

    def test_unknown_format_for_unrecognized_columns(self, tmp_path):
        """JSONL with random column names that match no known format is unknown."""
        rows = [
            {"question": "What color is the sky?", "answer": "Blue"},
            {"question": "What is 2+2?", "answer": "4"},
        ]
        path = tmp_path / "unknown.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.detected_format == "unknown"


# ---------------------------------------------------------------------------
# Sequence length estimation
# ---------------------------------------------------------------------------


class TestSequenceLengthEstimation:
    """Token count estimation via the chars/4 heuristic produces correct stats."""

    def test_known_text_lengths_produce_expected_token_estimates(self, tmp_path):
        """A 400-char string should estimate to 100 tokens (400 // 4)."""
        text_400_chars = "a" * 400
        text_800_chars = "b" * 800
        rows = [
            {"text": text_400_chars},
            {"text": text_800_chars},
        ]
        path = tmp_path / "known_lengths.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        stats = profile.seq_len_stats
        assert stats is not None

        # 400 chars // 4 = 100, 800 chars // 4 = 200
        assert stats.min == 100
        assert stats.max == 200

    def test_uniform_rows_produce_identical_percentiles(self, tmp_path):
        """If all rows have the same length, min == p50 == p95 == p99 == max."""
        text = "x" * 200  # 200 chars -> 50 tokens
        rows = [{"text": text} for _ in range(50)]
        path = tmp_path / "uniform.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        stats = profile.seq_len_stats
        assert stats.min == stats.max == 50
        assert stats.p50 == 50
        assert stats.p95 == 50
        assert stats.p99 == 50
        assert stats.mean == 50.0

    def test_p95_is_less_than_max_for_skewed_distribution(self, tmp_path):
        """p95 should reflect the 95th percentile, not the maximum value."""
        # 97 short rows + 3 long rows out of 100 total.
        # _compute_stats uses token_counts[int(n * 0.95)] = index 95.
        # The 3 long rows land at indices 97-99, so index 95 is still short.
        short_text = "a" * 100  # 25 tokens
        long_text = "b" * 10000  # 2500 tokens
        rows = [{"text": short_text}] * 97 + [{"text": long_text}] * 3
        path = tmp_path / "skewed.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        stats = profile.seq_len_stats
        assert stats.p95 < stats.max, (
            f"p95 ({stats.p95}) should be less than max ({stats.max}) when only 3% of rows are long"
        )

    def test_mean_reflects_actual_average(self, tmp_path):
        """Mean should be the arithmetic average of token estimates."""
        # 100 chars -> 25 tokens, 300 chars -> 75 tokens
        rows = [
            {"text": "a" * 100},
            {"text": "b" * 300},
        ]
        path = tmp_path / "mean_check.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        stats = profile.seq_len_stats
        assert stats.mean == pytest.approx(50.0), (
            f"Mean of [25, 75] should be 50.0, got {stats.mean}"
        )

    def test_very_short_text_gets_minimum_one_token(self, tmp_path):
        """Text shorter than 4 chars should still estimate to at least 1 token."""
        rows = [{"text": "Hi"}]  # 2 chars -> max(1, 2 // 4) = max(1, 0) = 1
        path = tmp_path / "short.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        stats = profile.seq_len_stats
        assert stats.min >= 1, "Token estimate should never be zero"

    def test_alpaca_counts_instruction_input_and_output(self, tmp_path):
        """Alpaca format sums chars from instruction + input + output fields."""
        rows = [
            {
                "instruction": "a" * 100,  # 100 chars
                "input": "b" * 200,  # 200 chars
                "output": "c" * 100,  # 100 chars
            },  # total: 400 chars -> 100 tokens
        ]
        path = tmp_path / "alpaca_count.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.seq_len_stats.min == 100

    def test_sharegpt_counts_all_conversation_turns(self, tmp_path):
        """ShareGPT format sums chars from all turn values."""
        rows = [
            {
                "conversations": [
                    {"from": "human", "value": "x" * 200},  # 200 chars
                    {"from": "gpt", "value": "y" * 200},  # 200 chars
                ]
            },  # total: 400 chars -> 100 tokens
        ]
        path = tmp_path / "sharegpt_count.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.seq_len_stats.min == 100


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


class TestFileHandling:
    """analyze_local validates file existence, extension, and content."""

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        """Asking to analyze a file that doesn't exist raises FileNotFoundError."""
        bogus_path = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError, match="not found"):
            analyze_local(bogus_path)

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        """A .csv file is not supported and should raise ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1,col2\na,b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            analyze_local(csv_path)

    def test_empty_jsonl_raises_value_error(self, tmp_path):
        """A JSONL file with no valid rows raises ValueError."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No valid rows"):
            analyze_local(path)

    def test_json_array_format_loads_correctly(self, tmp_path):
        """A .json file containing a top-level array is loaded as rows."""
        rows = [
            {"text": "First sentence."},
            {"text": "Second sentence."},
            {"text": "Third sentence."},
        ]
        path = tmp_path / "array.json"
        _write_json(path, rows)

        profile = analyze_local(path)
        assert profile.num_rows == 3
        assert profile.detected_format == "raw_text"

    def test_json_object_with_data_key_loads_correctly(self, tmp_path):
        """A .json file wrapping rows under a 'data' key is unwrapped."""
        inner_rows = [
            {"text": "Hello world."},
            {"text": "Goodbye world."},
        ]
        path = tmp_path / "wrapped.json"
        _write_json(path, {"data": inner_rows})

        profile = analyze_local(path)
        assert profile.num_rows == 2
        assert profile.detected_format == "raw_text"

    def test_json_object_with_rows_key_loads_correctly(self, tmp_path):
        """A .json file wrapping rows under a 'rows' key is unwrapped."""
        inner_rows = [
            {"instruction": "Do X.", "output": "Done."},
        ]
        path = tmp_path / "rows_key.json"
        _write_json(path, {"rows": inner_rows})

        profile = analyze_local(path)
        assert profile.num_rows == 1
        assert profile.detected_format == "alpaca"

    def test_jsonl_skips_blank_lines(self, tmp_path):
        """Blank lines in JSONL are silently skipped, not counted as rows."""
        content = '{"text": "line one"}\n\n\n{"text": "line two"}\n\n'
        path = tmp_path / "blanks.jsonl"
        path.write_text(content, encoding="utf-8")

        profile = analyze_local(path)
        assert profile.num_rows == 2

    def test_jsonl_skips_malformed_lines(self, tmp_path):
        """Malformed JSON lines are skipped; valid lines still load."""
        content = '{"text": "good line"}\nthis is not json\n{"text": "also good"}\n'
        path = tmp_path / "partial_bad.jsonl"
        path.write_text(content, encoding="utf-8")

        profile = analyze_local(path)
        assert profile.num_rows == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and special scenarios."""

    def test_single_row_produces_valid_stats(self, tmp_path):
        """A file with exactly one row should still produce a complete profile."""
        rows = [{"text": "a" * 400}]  # 400 chars -> 100 tokens
        path = tmp_path / "single.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        stats = profile.seq_len_stats
        assert stats is not None
        assert stats.min == stats.max == 100
        assert stats.mean == 100.0
        assert profile.num_rows == 1

    def test_sample_limit_caps_rows_read(self, tmp_path):
        """Setting sample_limit=5 means at most 5 rows are read from a large file."""
        rows = [{"text": f"row {i} " + "x" * 100} for i in range(100)]
        path = tmp_path / "large.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path, sample_limit=5)
        assert profile.num_rows == 5

    def test_profile_source_contains_file_path(self, tmp_path):
        """The returned profile's source field should reference the analyzed file."""
        rows = [{"text": "hello world"}]
        path = tmp_path / "source_check.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert "source_check.jsonl" in profile.source

    def test_num_rows_reflects_actual_rows_loaded(self, tmp_path):
        """num_rows should be the count of valid rows, not lines in the file."""
        rows = [{"text": f"sentence {i}"} for i in range(17)]
        path = tmp_path / "count.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.num_rows == 17

    def test_effective_seq_len_returns_p95(self, tmp_path):
        """DatasetProfile.effective_seq_len should be the p95 from seq_len_stats."""
        # 90 short rows + 10 long rows
        short = "a" * 200  # 50 tokens
        long = "b" * 2000  # 500 tokens
        rows = [{"text": short}] * 90 + [{"text": long}] * 10
        path = tmp_path / "effective.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.effective_seq_len == profile.seq_len_stats.p95

    def test_unknown_format_concatenates_all_string_values(self, tmp_path):
        """For unknown format, all string values in the row contribute to char count."""
        rows = [
            {"field_a": "a" * 200, "field_b": "b" * 200},  # 400 chars -> 100 tokens
        ]
        path = tmp_path / "unknown_count.jsonl"
        _write_jsonl(path, rows)

        profile = analyze_local(path)
        assert profile.detected_format == "unknown"
        assert profile.seq_len_stats.min == 100
