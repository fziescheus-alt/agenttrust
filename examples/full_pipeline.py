"""Full AgentTrust pipeline: PLAN → EXECUTE → QA → DELIVER.

Demonstrates the complete trust pipeline with a mock agent.
Replace the mock with your real LLM agent for production use.
"""

from agenttrust import Beipackzettel, TrustScore
from agenttrust.pipeline.pipeline import AgentPipeline
from agenttrust.qa.reviewer import review

# --- Step 1: Define your agent ---

call_count = 0

def my_research_agent(query: str) -> tuple[str, Beipackzettel]:
    """Mock research agent that returns a researched answer."""
    global call_count
    call_count += 1

    output = (
        f"## Research: {query}\n\n"
        f"Based on analysis of recent literature:\n\n"
        f"- Source: https://arxiv.org/abs/2506.04133 — TRiSM framework\n"
        f"- Reference: doi:10.1234/example — Calibration survey\n\n"
        f"Key findings:\n"
        f"- LLMs are overconfident 84% of the time (source: PMC/12249208)\n"
        f"- Sample consistency outperforms verbalized confidence estimation\n"
        f"- No production-ready trust framework exists yet\n\n"
        f"## Recommendation\n\n"
        f"Next step: Implement Budget-CoCoA for calibration.\n"
        f"Action item: Start with 3-sample consistency checks on critical claims.\n\n"
        f"Confidence: 72%\n"
        f"Beipackzettel: attached.\n\n"
        f"Risk: Literature review might fail to capture papers after Oct 2025.\n"
        f"Limitation: Coverage limited to English-language sources.\n"
        f"Uncertain about: Whether findings generalize to domain-specific models.\n"
        f"This is unclear and might not hold for all model families."
    )

    bpz = Beipackzettel(
        confidence=72.0,
        sources=["https://arxiv.org/abs/2506.04133", "PMC/12249208"],
        uncertainties=["Generalizability to domain-specific models"],
        risks=["Literature review may miss recent papers"],
        model="mock-agent-v1",
        agent_id="research-agent",
    )

    return output, bpz


# --- Step 2: Set up trust tracking ---

trust = TrustScore("research-agent", initial_score=15)
print(f"Initial trust: {trust}")
print(f"QA required: {trust.needs_qa()} (sample rate: {trust.qa_sample_rate():.0%})")
print()

# --- Step 3: Run the pipeline ---

pipeline = AgentPipeline(
    agent_fn=my_research_agent,
    trust_score=trust,
    tier=2,  # Standard quality threshold
)

result = pipeline.run("What calibration methods work for black-box LLMs?")

# --- Step 4: Inspect results ---

print("=" * 60)
print("PIPELINE RESULT")
print("=" * 60)
print(f"Delivered: {result.delivered}")
print(f"Iterations: {result.iterations}")
print()
print(result.beipackzettel)
print()

if result.review_result:
    r = result.review_result
    print(f"QA Score: {r.rubric_score.total}/{r.rubric_score.max_total}")
    print(f"Verdict: {r.verdict}")
    if r.issues:
        print(f"Issues: {', '.join(r.issues)}")
    print(f"Passes Tier 2: {r.rubric_score.passes(2)}")

print()
print(f"Updated trust: {trust}")
print(f"Trust summary: {trust.summary()}")
