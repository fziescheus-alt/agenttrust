"""Agent pipeline: plan → execute → review → deliver.

A structured pipeline that wraps any agent function with mandatory
quality gates. Every output gets a Beipackzettel and optional QA review
based on the agent's trust level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agenttrust.core.beipackzettel import Beipackzettel
from agenttrust.core.trust_score import TrustScore
from agenttrust.qa.reviewer import ReviewResult, review


@dataclass
class PipelineResult:
    """Result of a full pipeline run.

    Attributes:
        output: The agent's output text.
        beipackzettel: Mandatory metadata about the output.
        review_result: QA review result (None if skipped due to trust level).
        delivered: Whether the output passed QA and was delivered.
        iterations: How many plan→execute→review cycles were needed.
    """

    output: str
    beipackzettel: Beipackzettel
    review_result: ReviewResult | None = None
    delivered: bool = False
    iterations: int = 1


class AgentPipeline:
    """Structured pipeline: plan → execute → review → deliver.

    Wraps any agent function with trust infrastructure. The pipeline:
    1. Plans (optional planning step)
    2. Executes (calls the agent function)
    3. Reviews (QA based on trust level)
    4. Delivers (only if QA passes or trust is high enough)

    Args:
        agent_fn: Callable that takes a query and returns (output, beipackzettel).
        trust_score: The agent's TrustScore instance.
        tier: QA tier (1=quick, 2=standard, 3=deep).
        max_iterations: Max plan→execute→review cycles before giving up.

    Example::

        >>> def my_agent(query: str) -> tuple[str, Beipackzettel]:
        ...     return "Paris", Beipackzettel(confidence=90, sources=["wiki"])
        >>> ts = TrustScore("my-agent", initial_score=50)
        >>> pipeline = AgentPipeline(my_agent, ts)
        >>> result = pipeline.run("Capital of France?")
        >>> result.delivered
        True
    """

    def __init__(
        self,
        agent_fn: Callable[[str], tuple[str, Beipackzettel]],
        trust_score: TrustScore,
        tier: int = 2,
        max_iterations: int = 3,
    ) -> None:
        self.agent_fn = agent_fn
        self.trust_score = trust_score
        self.tier = tier
        self.max_iterations = max_iterations

    def run(self, query: str) -> PipelineResult:
        """Run the full pipeline for a query.

        Args:
            query: The task or question for the agent.

        Returns:
            PipelineResult with output, metadata, and delivery status.
        """
        for iteration in range(1, self.max_iterations + 1):
            # EXECUTE
            output, bpz = self.agent_fn(query)

            # REVIEW (based on trust level)
            qa_rate = self.trust_score.qa_sample_rate()
            review_result: ReviewResult | None = None

            if qa_rate >= 1.0:
                # Full QA
                review_result = review(output, tier=self.tier)
            elif qa_rate > 0:
                # Spot check — always review for pipeline runs
                review_result = review(output, tier=self.tier)

            # DELIVER decision
            if review_result is None:
                # Autonomous — no QA needed
                return PipelineResult(
                    output=output,
                    beipackzettel=bpz,
                    review_result=None,
                    delivered=True,
                    iterations=iteration,
                )

            if review_result.verdict == "PASS":
                self.trust_score.update(
                    stated_confidence=bpz.confidence,
                    outcome="good",
                )
                return PipelineResult(
                    output=output,
                    beipackzettel=bpz,
                    review_result=review_result,
                    delivered=True,
                    iterations=iteration,
                )

            if review_result.verdict == "FAIL" or iteration == self.max_iterations:
                self.trust_score.update(
                    stated_confidence=bpz.confidence,
                    outcome="bad",
                )
                return PipelineResult(
                    output=output,
                    beipackzettel=bpz,
                    review_result=review_result,
                    delivered=False,
                    iterations=iteration,
                )

            # REVISE — loop again

        # Should not reach here, but just in case
        return PipelineResult(
            output=output,
            beipackzettel=bpz,
            review_result=review_result,
            delivered=False,
            iterations=self.max_iterations,
        )
