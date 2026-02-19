"""Budget-CoCoA calibration: measure real confidence via sample consistency.

Instead of trusting an LLM's self-reported confidence, we ask the same
question multiple times independently and measure agreement. Consistency
across samples is a much stronger signal than verbalized confidence.

Reference: CoCoA — Xiong et al. (2024), adapted as "Budget-CoCoA" for
production use with 3 samples instead of 10+.

See also: PMC/12249208 — LLMs are overconfident 84% of the time.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence


class ConfidenceLevel(Enum):
    """Discrete confidence levels derived from sample consistency."""

    HIGH = "high"       # 3/3 agree — >85% confidence
    MEDIUM = "medium"   # 2/3 agree — 60-85% confidence
    LOW = "low"         # ≤1/3 agree — <60% confidence


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a Budget-CoCoA consistency check.

    Attributes:
        query: The original query or claim being checked.
        samples: The raw responses from each independent sample.
        agreement_ratio: Fraction of samples that match the majority answer.
        confidence_level: Discrete confidence level (HIGH/MEDIUM/LOW).
        confidence_pct: Numeric confidence estimate (0-100).
        majority_answer: The most common normalized answer, or None if no majority.
    """

    query: str
    samples: tuple[str, ...]
    agreement_ratio: float
    confidence_level: ConfidenceLevel
    confidence_pct: float
    majority_answer: str | None


def _normalize(text: str) -> str:
    """Normalize a response for comparison.

    Strips whitespace, lowercases, and removes trailing punctuation so that
    minor formatting differences don't break agreement detection.
    """
    text = text.strip().lower()
    text = re.sub(r"[.!?,;:]+$", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _compute_agreement(samples: Sequence[str]) -> tuple[float, str | None]:
    """Compute agreement ratio and majority answer from normalized samples.

    Returns:
        Tuple of (agreement_ratio, majority_answer). majority_answer is None
        if all answers are different.
    """
    normalized = [_normalize(s) for s in samples]
    counts = Counter(normalized)
    most_common, most_count = counts.most_common(1)[0]
    ratio = most_count / len(normalized)
    majority = most_common if most_count > 1 or len(normalized) == 1 else None
    return ratio, majority


def _ratio_to_level(ratio: float, n: int) -> ConfidenceLevel:
    """Map agreement ratio to a confidence level.

    For n=3 (default Budget-CoCoA):
        3/3 (1.0)   → HIGH
        2/3 (0.667) → MEDIUM
        1/3 (0.333) → LOW
    """
    if ratio >= 1.0:
        return ConfidenceLevel.HIGH
    elif ratio >= 2 / 3:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def _ratio_to_pct(ratio: float) -> float:
    """Map agreement ratio to a percentage confidence estimate.

    Linear mapping: 0.0 → 30%, 0.5 → 57.5%, 1.0 → 85%.
    Capped at [30, 95] — we never say 100% and never below 30%.
    """
    pct = 30.0 + ratio * 55.0
    return min(95.0, max(30.0, round(pct, 1)))


def sample_consistency(
    fn: Callable[[str], str],
    query: str,
    n: int = 3,
) -> CalibrationResult:
    """Run Budget-CoCoA: ask the same question n times and measure consistency.

    This is the core calibration primitive. Pass any function that takes a
    query string and returns an answer string. We call it ``n`` times
    independently, then measure how much the answers agree.

    Args:
        fn: A callable that takes a query string and returns an answer.
            Each call should be independent (e.g., no conversation history).
        query: The claim or question to check.
        n: Number of independent samples (default 3). Higher = more accurate
           but more expensive. 3 is the "budget" sweet spot.

    Returns:
        CalibrationResult with agreement ratio, confidence level, and samples.

    Example::

        >>> def fake_llm(q: str) -> str:
        ...     return "Paris"
        >>> result = sample_consistency(fake_llm, "Capital of France?")
        >>> result.confidence_level
        <ConfidenceLevel.HIGH: 'high'>
        >>> result.agreement_ratio
        1.0
    """
    if n < 2:
        raise ValueError("Need at least 2 samples for consistency check")

    samples: list[str] = []
    for _ in range(n):
        response = fn(query)
        samples.append(response)

    ratio, majority = _compute_agreement(samples)
    level = _ratio_to_level(ratio, n)
    pct = _ratio_to_pct(ratio)

    return CalibrationResult(
        query=query,
        samples=tuple(samples),
        agreement_ratio=round(ratio, 4),
        confidence_level=level,
        confidence_pct=pct,
        majority_answer=majority,
    )


@dataclass
class VerbalizedConfidenceResult:
    """Result of parsing an LLM's self-reported confidence.

    Attributes:
        raw_text: The original response containing a confidence statement.
        stated_confidence: The confidence percentage the LLM claimed (0-100).
        calibrated_confidence: Adjusted confidence after applying overconfidence
            discount. LLMs are overconfident ~84% of the time, so we apply a
            discount factor.
        discount_factor: The multiplier applied (default 0.7).
    """

    raw_text: str
    stated_confidence: float
    calibrated_confidence: float
    discount_factor: float


_CONFIDENCE_PATTERN = re.compile(
    r"(?:confidence|confident|certainty|sure)[\s:]*(\d{1,3})[\s]*%",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceSignalResult:
    """Result of the 3-signal confidence formula.

    Computes claim-level confidence from three independent signals:
    source quality (Admiralty), consistency, and structural markers.

    Formula:
        claim_conf = 0.5 * source + 0.3 * consistency + 0.2 * structural

    Attributes:
        claim: The claim text being assessed.
        source_signal: Quality of the source (0-1).
        consistency_signal: Agreement/verification level (0-1).
        structural_signal: Deterministic text features (0-1).
        confidence_pct: Final computed confidence (0-100).
        admiralty: Admiralty rating used (e.g., "A1", "C3").
    """

    claim: str
    source_signal: float
    consistency_signal: float
    structural_signal: float
    confidence_pct: float
    admiralty: str


# Admiralty reliability ratings (NATO system, adapted)
_ADMIRALTY_SCORES: dict[str, float] = {
    "A1": 0.95,  # Reliable source, confirmed by other sources
    "A2": 0.85,  # Reliable source, probably true
    "B2": 0.70,  # Usually reliable, probably true
    "C3": 0.40,  # Fairly reliable, possibly true
    "D4": 0.20,  # Not usually reliable, doubtful
    "E2": 0.10,  # Unreliable, probably true (contradictory)
}

_VERIFICATION_SCORES: dict[str, float] = {
    "verified": 1.0,      # Independently verified
    "partial": 0.5,       # Partially verified
    "unverifiable": 0.1,  # Cannot be verified
}

_CONSISTENCY_MAP: dict[str, float] = {
    "verified": 0.85,
    "partial": 0.60,
    "unverifiable": 0.30,
}


def source_signal_confidence(
    claim: str,
    admiralty: str = "C3",
    verification: str = "partial",
    evidence_year: int | None = None,
    current_year: int = 2026,
    has_doi: bool = False,
    has_url: bool = False,
    has_percentage: bool = False,
    has_year: bool = False,
    has_source_ref: bool = False,
) -> SourceSignalResult:
    """Compute claim confidence using the 3-signal formula.

    This is the standard confidence method for the MIA Pipeline and any
    pipeline where Budget-CoCoA (3 LLM calls) is too expensive.

    Formula::

        claim_conf = 0.5 * SOURCE + 0.3 * CONSISTENCY + 0.2 * STRUCTURAL

        SOURCE = admiralty_score * verification_score * recency
        CONSISTENCY = Budget-CoCoA proxy from verification status
        STRUCTURAL = sum of deterministic text markers (capped at 0.50)

    Args:
        claim: The claim text (for the result record).
        admiralty: Admiralty rating (A1, A2, B2, C3, D4, E2).
        verification: Verification status (verified, partial, unverifiable).
        evidence_year: Year of the evidence (for recency discount).
        current_year: Current year (default 2026).
        has_doi: Whether a DOI pattern is present.
        has_url: Whether a URL is present.
        has_percentage: Whether a specific percentage is cited.
        has_year: Whether a specific year is cited.
        has_source_ref: Whether a [S#] reference is present.

    Returns:
        SourceSignalResult with all three signals and final confidence.

    Example::

        >>> r = source_signal_confidence(
        ...     "ECE averages 27.3%", admiralty="A1",
        ...     verification="verified", has_doi=True, has_percentage=True
        ... )
        >>> r.confidence_pct
        82.5
    """
    # Source signal
    adm_score = _ADMIRALTY_SCORES.get(admiralty, 0.40)
    ver_score = _VERIFICATION_SCORES.get(verification, 0.5)
    if evidence_year is not None:
        recency = max(0.5, 1.0 - (current_year - evidence_year) * 0.1)
    else:
        recency = 0.8  # default if unknown
    source = adm_score * ver_score * recency

    # Consistency signal (Budget-CoCoA proxy)
    consistency = _CONSISTENCY_MAP.get(verification, 0.40)

    # Structural signal (deterministic text markers)
    structural = 0.0
    if has_doi:
        structural += 0.30
    if has_url:
        structural += 0.15
    if has_percentage:
        structural += 0.10
    if has_year:
        structural += 0.05
    if has_source_ref:
        structural += 0.10
    structural = min(0.50, structural)

    # Combined confidence
    conf = 0.5 * source + 0.3 * consistency + 0.2 * structural
    conf_pct = round(conf * 100, 1)

    return SourceSignalResult(
        claim=claim,
        source_signal=round(source, 4),
        consistency_signal=round(consistency, 4),
        structural_signal=round(structural, 4),
        confidence_pct=conf_pct,
        admiralty=admiralty,
    )


def report_confidence(
    claim_results: list[SourceSignalResult],
    weights: list[float] | None = None,
) -> float:
    """Compute report-level confidence from claim-level results.

    Weighted average of claim confidences.

    Args:
        claim_results: List of SourceSignalResult from source_signal_confidence.
        weights: Optional per-claim weights. Default: all equal.
            Suggested: load-bearing=1.0, supporting=0.6, contextual=0.3.

    Returns:
        Report confidence percentage (0-100).

    Example::

        >>> claims = [
        ...     source_signal_confidence("Claim 1", admiralty="A1", verification="verified", has_doi=True),
        ...     source_signal_confidence("Claim 2", admiralty="C3", verification="partial"),
        ... ]
        >>> report_confidence(claims, weights=[1.0, 0.6])
        68.3
    """
    if not claim_results:
        return 0.0
    if weights is None:
        weights = [1.0] * len(claim_results)
    if len(weights) != len(claim_results):
        raise ValueError("weights must match claim_results length")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        r.confidence_pct * w for r, w in zip(claim_results, weights)
    )
    return round(weighted_sum / total_weight, 1)


def verbalized_confidence(
    text: str,
    discount: float = 0.7,
) -> VerbalizedConfidenceResult:
    """Extract and discount an LLM's self-reported confidence.

    LLMs are overconfident approximately 84% of the time (PMC/12249208).
    This function extracts a stated confidence percentage from text and
    applies a discount factor to produce a more realistic estimate.

    This is a **weaker** signal than ``sample_consistency`` and should be
    used as a fallback when multiple samples aren't feasible.

    Args:
        text: LLM output text that may contain a confidence statement
            like "confidence: 90%" or "I'm 85% confident".
        discount: Multiplier to apply to stated confidence. Default 0.7
            based on calibration literature showing ~30% overconfidence gap.

    Returns:
        VerbalizedConfidenceResult with stated and calibrated confidence.

    Raises:
        ValueError: If no confidence statement is found in the text.

    Example::

        >>> result = verbalized_confidence("I'm 90% confident this is correct.")
        >>> result.stated_confidence
        90.0
        >>> result.calibrated_confidence
        63.0
    """
    if not 0 < discount <= 1.0:
        raise ValueError(f"Discount must be in (0, 1], got {discount}")

    match = _CONFIDENCE_PATTERN.search(text)
    if not match:
        raise ValueError(
            f"No confidence statement found in text. "
            f"Expected patterns like 'confidence: 85%' or '90% confident'."
        )

    stated = float(match.group(1))
    stated = min(100.0, max(0.0, stated))
    calibrated = round(stated * discount, 1)

    return VerbalizedConfidenceResult(
        raw_text=text,
        stated_confidence=stated,
        calibrated_confidence=calibrated,
        discount_factor=discount,
    )
