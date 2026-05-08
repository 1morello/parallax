"""
SHAP-based explanations for plan-grant matches.
Shows which words/phrases in the plan contributed most to the similarity score.
"""

import numpy as np

from src.retrieval.embedder import Embedder
from src.preprocessing.text_cleaner import clean_text
from src.data_collection.loader import load_grants, get_grant_by_id


class ShapExplainer:

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()

    def explain(self, plan_text: str, grant_id: str,
                n_samples: int = 100, top_k: int = 10) -> dict:
        """
        Approximate SHAP values by masking words and measuring
        the effect on similarity.

        Not true Shapley values (that's combinatorially insane for text),
        but a reasonable perturbation-based approximation.

        Returns dict with:
            - "features": list of (word, contribution) tuples, sorted by |contribution|
            - "base_score": similarity with full text
            - "grant_id": which grant we're explaining
        """
        cleaned = clean_text(plan_text)
        words = cleaned.split()
        grant = get_grant_by_id(grant_id)

        grant_vec = self.embedder.embed_text(clean_text(grant.description))
        full_vec = self.embedder.embed_long_text(plan_text)
        base_score = float(np.dot(full_vec, grant_vec))

        # for each word, estimate its marginal contribution
        # by randomly masking it across multiple subsets
        contributions = np.zeros(len(words))
        rng = np.random.default_rng(42)

        for _ in range(n_samples):
            # random mask: each word has 50% chance of being included
            mask = rng.random(len(words)) > 0.5

            # score with mask
            masked_text = " ".join(w for w, m in zip(words, mask) if m)
            if not masked_text.strip():
                continue
            masked_vec = self.embedder.embed_text(masked_text)
            score_with = float(np.dot(masked_vec, grant_vec))

            # score without each word that was included
            for i in range(len(words)):
                if mask[i]:
                    mask_without = mask.copy()
                    mask_without[i] = False
                    text_without = " ".join(w for w, m in zip(words, mask_without) if m)
                    if not text_without.strip():
                        continue
                    vec_without = self.embedder.embed_text(text_without)
                    score_without = float(np.dot(vec_without, grant_vec))
                    contributions[i] += (score_with - score_without)

        # average contributions
        contributions /= n_samples

        # group by word and sort
        word_scores = list(zip(words, contributions))
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "features": word_scores[:top_k],
            "base_score": base_score,
            "grant_id": grant_id,
        }

    def format_explanation(self, result: dict) -> str:
        lines = [f"SHAP explanation for grant {result['grant_id']} (base score: {result['base_score']:.3f})\n"]

        for word, contrib in result["features"]:
            direction = "+" if contrib > 0 else ""
            bar = "█" * int(abs(contrib) * 200)
            lines.append(f"  {direction}{contrib:.4f}  {bar}  {word}")

        return "\n".join(lines)


if __name__ == "__main__":
    from src.data_collection.loader import load_plans

    explainer = ShapExplainer()
    plan = load_plans()[0]  # Frascati

    # explain top grant match
    result = explainer.explain(plan.full_text, "ERDF-2026-IT-URBAN", n_samples=50, top_k=8)
    print(explainer.format_explanation(result))
