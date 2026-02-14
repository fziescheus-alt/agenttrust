# AgentTrust

**Trust infrastructure for AI agents. Because confidence ≠ correctness.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Website](https://img.shields.io/badge/website-ainaryventures.com-gold.svg)](https://ainaryventures.com)

---

## The Problem

LLMs are overconfident **84% of the time** ([PMC/12249208](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12249208/), Feb 2025). When an AI agent says "I'm 95% confident," it's usually wrong about how right it is.

Every agent framework lets you **build** agents. None of them help you **trust** them.

- **LangChain** gives you chains. No trust scores.
- **CrewAI** gives you teams. No calibration.
- **AutoGen** gives you conversations. No accountability.

AgentTrust fills the gap: calibration, trust scoring, and mandatory metadata for every agent output.

## The Solution

Three primitives that work with any agent framework:

| Primitive | What it does |
|-----------|-------------|
| **Calibration** | Ask the same question 3x independently. Measure agreement. Real confidence, not vibes. |
| **Trust Scores** | Track agent accuracy over time. Honest uncertainty → more autonomy. Overconfidence → more oversight. |
| **Beipackzettel** | Mandatory metadata on every output: confidence, sources, uncertainties, risks. The nutrition label for AI. |

## Quick Start

```bash
pip install agenttrust
```

```python
from agenttrust import sample_consistency

# Wrap any LLM function
result = sample_consistency(my_llm, "What causes inflation?", n=3)

print(result.confidence_level)   # HIGH / MEDIUM / LOW
print(result.confidence_pct)     # 85.0
print(result.agreement_ratio)    # 1.0 (3/3 agreed)
```

That's it. Five lines to go from "the model said so" to "here's how much we should trust it."

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     AgentTrust                          │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  PLAN    │→ │   EXECUTE    │→ │     REVIEW       │  │
│  │          │  │              │  │                   │  │
│  │ Define   │  │ Agent runs   │  │ 8-dim rubric     │  │
│  │ task +   │  │ task with    │  │ scores output    │  │
│  │ tier     │  │ Beipackzettel│  │ 0-16 points      │  │
│  └──────────┘  └──────────────┘  └────────┬──────────┘  │
│                                           │             │
│                              ┌────────────┴──────┐      │
│                              │     DELIVER       │      │
│                              │                   │      │
│                              │ Pass → ship it    │      │
│                              │ Revise → loop     │      │
│                              │ Fail → flag it    │      │
│                              └───────────────────┘      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              TRUST SCORE (per agent)             │    │
│  │                                                  │    │
│  │  0-30: QA reviews all    │  61-80: spot-checks  │    │
│  │  31-60: QA reviews flags │  81+: autonomous     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              CALIBRATION                         │    │
│  │                                                  │    │
│  │  Budget-CoCoA: 3 independent samples → agreement │    │
│  │  3/3 agree → HIGH    2/3 → MEDIUM    else → LOW │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Core Concepts

### Budget-CoCoA Calibration

Instead of trusting an LLM's self-reported confidence, ask the same question 3 times independently and measure agreement. Consistency across samples is a stronger signal than "I'm 90% sure."

```python
from agenttrust import sample_consistency

result = sample_consistency(my_llm, "Is Berlin the capital of Germany?")
# result.confidence_level → ConfidenceLevel.HIGH
# result.agreement_ratio → 1.0
# result.confidence_pct   → 85.0
```

Based on [CoCoA (Xiong et al., 2024)](https://arxiv.org/abs/2407.08461), adapted for production with 3 samples instead of 10+.

### Trust Scores

Every agent builds a reputation. Good calibration earns trust. Overconfidence destroys it.

```python
from agenttrust import TrustScore

trust = TrustScore("my-agent")

trust.update(stated_confidence=85, outcome="good")    # +1
trust.update(stated_confidence=95, outcome="bad")      # -3 (overconfident!)
trust.update(stated_confidence=50, outcome="flagged_real")  # +2 (honest!)

print(trust.score)        # 0
print(trust.trust_level)  # TrustLevel.UNTRUSTED
print(trust.needs_qa())   # True
```

### Beipackzettel (Package Insert)

Every output ships with mandatory metadata. Like a nutrition label, but for AI.

```python
from agenttrust import Beipackzettel

bpz = Beipackzettel(
    confidence=72.0,
    sources=["https://arxiv.org/abs/2506.04133", "PMC/12249208"],
    uncertainties=["May not generalize to domain-specific models"],
    risks=["Literature review may miss papers after Oct 2025"],
    model="claude-sonnet-4-20250514",
)

print(bpz.risk_level)   # 'medium'
print(bpz.is_grounded)  # True
print(bpz)              # Human-readable summary
```

### QA Review

Score any output against an 8-dimension rubric (0-2 each, max 16):

```python
from agenttrust.qa.reviewer import review

result = review(agent_output, tier=2)
print(result.rubric_score.total)  # e.g., 13/16
print(result.verdict)             # PASS / REVISE / FAIL
```

**Dimensions:** Accuracy · Completeness · Sources · Clarity · Epistemic Honesty · Actionability · Calibration · Risk Awareness

### Full Pipeline

```python
from agenttrust import TrustScore, Beipackzettel
from agenttrust.pipeline.pipeline import AgentPipeline

pipeline = AgentPipeline(
    agent_fn=my_agent,        # (query) → (output, Beipackzettel)
    trust_score=trust,
    tier=2,
)

result = pipeline.run("Research calibration methods for LLMs")
# result.delivered → True/False
# result.review_result → QA scores
# result.beipackzettel → metadata
```

## Works With

AgentTrust is framework-agnostic. Use it with:

- **Raw OpenAI / Anthropic** — wrap your API calls
- **LangChain** — middleware coming in v0.2
- **CrewAI** — wrap agent outputs
- **AutoGen** — wrap conversation results
- **Any callable** — if it takes a string and returns a string, it works

```python
# OpenAI example
from agenttrust.integrations.openai_provider import create_calibrated_fn

fn = create_calibrated_fn(model="gpt-4o-mini")
result = sample_consistency(fn, "What is quantum computing?")
```

## Installation

```bash
# Core (no dependencies)
pip install agenttrust

# With OpenAI support
pip install agenttrust[openai]

# Development
pip install agenttrust[dev]
```

Requires Python ≥ 3.10.

## Research Background

This framework builds on:

- **LLM Overconfidence:** LLMs are miscalibrated 84% of the time ([PMC/12249208](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12249208/))
- **TRiSM for Agentic AI:** Trust, Risk, and Security Management framework ([Raza et al., 2025](https://arxiv.org/abs/2506.04133))
- **Confidence Gap:** "Mind the Confidence Gap" — systematic overconfidence in LLM self-assessment ([arXiv:2502.11028](https://arxiv.org/abs/2502.11028))
- **CoCoA:** Consistency-based calibration ([Xiong et al., 2024](https://arxiv.org/abs/2407.08461))

No production-ready trust framework for AI agents existed before AgentTrust. The academic research is active; the tooling was missing.

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

```bash
# Run tests
pip install -e ".[dev]"
pytest

# Lint
ruff check .
```

## License

[Apache 2.0](LICENSE) — use it, fork it, build on it.

---

*Built by [Florian Ziesche](https://github.com/florianziesche). Born from the realization that we build agents but never ask: "should we trust this?"*
