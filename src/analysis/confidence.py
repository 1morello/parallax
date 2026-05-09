"""
Confidence calibration utilities.

Adds a meta-layer on top of raw similarity scores:
how much should the user trust each match result?

Most of the actual calibration logic lives in Matcher._calibrate(),
but this module provides standalone analysis and visualization helpers.
"""

import numpy as np

from src.models import MatchResult
from config.settings import (
    CONFIDENCE_GAP_THRESHOLD,
    CONFIDENCE_CLUSTER_RANGE,
)


def analyze_score_distribution(results: list[MatchResult]) -> dict:
    """
    Analyze the distribution of match scores to assess
    overall result quality.
    """
    if not results:
        return {"status": "no results"}

    scores = np.array([r.similarity_score for r in results])

    top_score = float(scores[0])
    score_range = float(scores.max() - scores.min())
    score_std = float(scores.std())

    # how many results are clustered near the top?
    near_top = sum(1 for s in scores if abs(s - top_score) <= CONFIDENCE_CLUSTER_RANGE)

    # gap between #1 and #2
    top_gap = float(scores[0] - scores[1]) if len(scores) > 1 else top_score

    if top_score > 0.75 and top_gap > CONFIDENCE_GAP_THRESHOLD:
        quality = "strong"
        summary = "Clear top match with high confidence."
    elif score_std < 0.03:
        quality = "ambiguous"
        summary = "Scores are very close together — hard to distinguish best match."
    elif top_score < 0.5:
        quality = "weak"
        summary = "Low similarity overall — plan may not align well with available grants."
    else:
        quality = "moderate"
        summary = "Reasonable matches found, but no single standout."

    return {
        "quality": quality,
        "summary": summary,
        "top_score": top_score,
        "top_gap": top_gap,
        "score_std": float(score_std),
        "score_range": score_range,
        "near_top_cluster": near_top,
        "n_high": sum(1 for r in results if r.confidence == "high"),
        "n_medium": sum(1 for r in results if r.confidence == "medium"),
        "n_low": sum(1 for r in results if r.confidence == "low"),
    }


def format_confidence_report(results: list[MatchResult]) -> str:
    analysis = analyze_score_distribution(results)

    lines = [
        f"Confidence analysis: {analysis['quality'].upper()}\n",
        f"  {analysis['summary']}\n",
        f"  Top score: {analysis['top_score']:.3f}",
        f"  Gap to #2: {analysis['top_gap']:.3f}",
        f"  Score spread (std): {analysis['score_std']:.3f}",
        f"  Results near top: {analysis['near_top_cluster']}",
        f"  Confidence breakdown: "
        f"{analysis['n_high']} high, {analysis['n_medium']} medium, {analysis['n_low']} low",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    from src.retrieval.matcher import Matcher

    matcher = Matcher()

    from src.data_collection.loader import load_plans
    for plan in load_plans():
        results = matcher.match_plan(plan.id, top_k=10)
        print(f"\n── {plan.municipality.upper()} ──")
        print(format_confidence_report(results))
