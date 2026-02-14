"""AgentTrust Quick Start — Calibrate a claim in 5 lines."""

from agenttrust import sample_consistency

# Any function that answers a question — swap in your LLM
answers = iter(["Paris", "Paris", "Paris"])
result = sample_consistency(lambda q: next(answers), "What is the capital of France?")

print(f"Confidence: {result.confidence_level.value} ({result.confidence_pct}%)")
print(f"Agreement:  {result.agreement_ratio:.0%} ({len(result.samples)} samples)")
print(f"Answer:     {result.majority_answer}")
