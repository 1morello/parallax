"""
Semantic matching: plan vector → ranked list of grants.
Includes confidence calibration (how much to trust each result).
"""

from datetime import date

import numpy as np

from config.settings import (
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    CONFIDENCE_GAP_THRESHOLD,
    CONFIDENCE_CLUSTER_RANGE,
    CONFIDENCE_CLUSTER_MAX_SIZE,
)
from src.models import MatchResult
from src.data_collection.loader import load_grants, get_grant_by_id, get_plan_by_id
from src.retrieval.embedder import Embedder, load_grant_embeddings, load_plan_embeddings


class Matcher:

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.grant_embeddings = load_grant_embeddings()
        self.grants = load_grants()

    def match_plan(self, plan_id: str, top_k: int = TOP_K_RESULTS,
                   deadline_filter: date | None = None) -> list[MatchResult]:
        """Match a pre-embedded plan against all grants."""
        plan_embs = load_plan_embeddings()
        idx = plan_embs["ids"].index(plan_id)
        return self._rank(plan_embs["vectors"][idx], plan_id, top_k, deadline_filter)

    def match_text(self, text: str, plan_id: str = "custom",
                   top_k: int = TOP_K_RESULTS,
                   deadline_filter: date | None = None) -> list[MatchResult]:
        """Match arbitrary text — used by the What-If explorer."""
        vec = self.embedder.embed_long_text(text)
        return self._rank(vec, plan_id, top_k, deadline_filter)

    def _rank(self, plan_vector: np.ndarray, plan_id: str,
              top_k: int, deadline_filter: date | None) -> list[MatchResult]:

        grant_ids = self.grant_embeddings["ids"]
        grant_vectors = self.grant_embeddings["vectors"]

        # vectors are L2-normalized, so dot = cosine similarity
        scores = np.dot(grant_vectors, plan_vector)

        scored = sorted(zip(grant_ids, scores), key=lambda x: x[1], reverse=True)

        if deadline_filter:
            scored = [(gid, s) for gid, s in scored
                      if self._deadline_ok(gid, deadline_filter)]

        scored = [(gid, s) for gid, s in scored if s >= SIMILARITY_THRESHOLD]
        top = scored[:top_k]

        all_scores = [s for _, s in scored]
        results = []

        for rank, (grant_id, score) in enumerate(top, start=1):
            grant = get_grant_by_id(grant_id, self.grants)
            plan = get_plan_by_id(plan_id)

            shared = []
            if grant and plan:
                shared = sorted(set(grant.themes) & set(plan.themes))

            results.append(MatchResult(
                plan_id=plan_id,
                grant_id=grant_id,
                similarity_score=round(float(score), 4),
                rank=rank,
                confidence=self._calibrate(score, rank, all_scores),
                shared_themes=shared,
            ))

        return results

    def _deadline_ok(self, grant_id: str, cutoff: date) -> bool:
        grant = get_grant_by_id(grant_id, self.grants)
        if not grant or not grant.deadline:
            return True
        return grant.deadline >= cutoff

    def _calibrate(self, score: float, rank: int, all_scores: list[float]) -> str:
        """
        Confidence heuristic. Not scientific — just useful.
        High = clear winner or very strong score.
        Low = too many similar scores clustered together, or near threshold.
        """
        if len(all_scores) < 2:
            return "high" if score > 0.6 else "medium"

        gap = score - all_scores[rank] if rank < len(all_scores) else score

        cluster_size = sum(1 for s in all_scores
                          if abs(s - score) <= CONFIDENCE_CLUSTER_RANGE)

        if score > 0.75 and gap > CONFIDENCE_GAP_THRESHOLD:
            return "high"
        if cluster_size > CONFIDENCE_CLUSTER_MAX_SIZE:
            return "low"
        if score < SIMILARITY_THRESHOLD + 0.1:
            return "low"
        if gap > CONFIDENCE_GAP_THRESHOLD:
            return "high"
        return "medium"


def format_results(results: list[MatchResult], grants: list | None = None) -> str:
    """Pretty-print for terminal. Not used in production — just for debugging."""
    grants = grants or load_grants()
    lines = []

    for r in results:
        g = get_grant_by_id(r.grant_id, grants)
        name = g.name if g else r.grant_id
        prog = g.programme if g else "?"
        budget = f"€{g.budget_eur:,}" if g and g.budget_eur else "N/A"
        dl = str(g.deadline) if g and g.deadline else "N/A"
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[r.confidence]

        lines.append(
            f"  #{r.rank}  {icon} {r.similarity_score:.3f}  {name}\n"
            f"       {prog} | {budget} | deadline {dl}\n"
            f"       themes: {', '.join(r.shared_themes) or '-'}"
        )

    return "\n\n".join(lines)


if __name__ == "__main__":
    matcher = Matcher()

    for pid in ["plan-frascati", "plan-montescaglioso", "plan-bolzano"]:
        plan = get_plan_by_id(pid)
        print(f"\n{'='*60}")
        print(f"  {plan.municipality.upper()} ({plan.region})")
        print(f"{'='*60}")
        print(format_results(matcher.match_plan(pid, top_k=5)))
