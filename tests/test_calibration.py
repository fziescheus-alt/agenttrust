"""Tests for Budget-CoCoA calibration."""

import pytest

from agenttrust.core.calibration import (
    CalibrationResult,
    ConfidenceLevel,
    VerbalizedConfidenceResult,
    sample_consistency,
    verbalized_confidence,
    _normalize,
    _compute_agreement,
)


# --- _normalize ---

class TestNormalize:
    def test_strips_whitespace(self) -> None:
        assert _normalize("  hello  ") == "hello"

    def test_lowercases(self) -> None:
        assert _normalize("PARIS") == "paris"

    def test_removes_trailing_punctuation(self) -> None:
        assert _normalize("Paris.") == "paris"
        assert _normalize("Yes!") == "yes"

    def test_collapses_whitespace(self) -> None:
        assert _normalize("hello   world") == "hello world"


# --- _compute_agreement ---

class TestComputeAgreement:
    def test_full_agreement(self) -> None:
        ratio, majority = _compute_agreement(["Paris", "Paris", "Paris"])
        assert ratio == 1.0
        assert majority == "paris"

    def test_partial_agreement(self) -> None:
        ratio, majority = _compute_agreement(["Paris", "Paris", "London"])
        assert abs(ratio - 2 / 3) < 0.01
        assert majority == "paris"

    def test_no_agreement(self) -> None:
        ratio, majority = _compute_agreement(["Paris", "London", "Berlin"])
        assert abs(ratio - 1 / 3) < 0.01
        assert majority is None

    def test_case_insensitive(self) -> None:
        ratio, _ = _compute_agreement(["Paris", "paris", "PARIS"])
        assert ratio == 1.0

    def test_punctuation_insensitive(self) -> None:
        ratio, _ = _compute_agreement(["Paris.", "Paris!", "Paris"])
        assert ratio == 1.0


# --- sample_consistency ---

class TestSampleConsistency:
    def test_high_confidence(self) -> None:
        answers = iter(["Paris", "Paris", "Paris"])
        result = sample_consistency(lambda q: next(answers), "capital of France?")
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.agreement_ratio == 1.0
        assert result.majority_answer == "paris"

    def test_medium_confidence(self) -> None:
        answers = iter(["Paris", "Paris", "Lyon"])
        result = sample_consistency(lambda q: next(answers), "capital of France?")
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_low_confidence(self) -> None:
        answers = iter(["Paris", "Lyon", "Marseille"])
        result = sample_consistency(lambda q: next(answers), "capital of France?")
        assert result.confidence_level == ConfidenceLevel.LOW

    def test_custom_n(self) -> None:
        answers = iter(["A", "A", "A", "A", "B"])
        result = sample_consistency(lambda q: next(answers), "test?", n=5)
        assert len(result.samples) == 5
        assert result.agreement_ratio == 0.8

    def test_n_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            sample_consistency(lambda q: "x", "test?", n=1)

    def test_returns_calibration_result(self) -> None:
        result = sample_consistency(lambda q: "yes", "test?")
        assert isinstance(result, CalibrationResult)
        assert result.query == "test?"
        assert 30 <= result.confidence_pct <= 95


# --- verbalized_confidence ---

class TestVerbalizedConfidence:
    def test_extracts_confidence(self) -> None:
        result = verbalized_confidence("I'm 90% confident this is correct.")
        assert result.stated_confidence == 90.0

    def test_applies_discount(self) -> None:
        result = verbalized_confidence("Confidence: 80%", discount=0.7)
        assert result.calibrated_confidence == 56.0
        assert result.discount_factor == 0.7

    def test_custom_discount(self) -> None:
        result = verbalized_confidence("confidence: 100%", discount=0.5)
        assert result.calibrated_confidence == 50.0

    def test_no_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="No confidence statement"):
            verbalized_confidence("I think the answer is Paris.")

    def test_invalid_discount_raises(self) -> None:
        with pytest.raises(ValueError, match="Discount"):
            verbalized_confidence("confidence: 80%", discount=0.0)
        with pytest.raises(ValueError, match="Discount"):
            verbalized_confidence("confidence: 80%", discount=1.5)

    def test_various_formats(self) -> None:
        # "confidence: X%"
        r1 = verbalized_confidence("confidence: 85%")
        assert r1.stated_confidence == 85.0

        # "X% confident"
        r2 = verbalized_confidence("I am 75% confident")
        assert r2.stated_confidence == 75.0

        # "certainty: X%"
        r3 = verbalized_confidence("certainty: 60%")
        assert r3.stated_confidence == 60.0
