"""
Bias detection: does the system treat all municipalities fairly?

Compares matching scores across plans with different writing styles,
city sizes, and regions to detect systematic disparities.
"""

import numpy as np

from src.data_collection.loader import load_plans, load_grants
from src.retrieval.embedder import Embedder
from src.preprocessing.text_cleaner import clean_text


class BiasDetector:

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()

    def run_style_bias_test(self) -> dict:
        """
        Test: do formal vs informal plans get systematically different scores?

        Takes the same core meaning, compares average similarity scores
        between formal-style and informal-style plans against all grants.
        """
        plans = load_plans()
        grants = load_grants()

        grant_vecs = self.embedder.embed_texts(
            [clean_text(g.description) for g in grants]
        )

        results = []
        for plan in plans:
            plan_vec = self.embedder.embed_long_text(plan.full_text)
            scores = np.dot(grant_vecs, plan_vec)

            # we tagged style in the plans json
            # infer from the data: plans with "Cari concittadini" or
            # "Cosa vogliamo fare" are informal
            style = _detect_style(plan.full_text)

            results.append({
                "municipality": plan.municipality,
                "region": plan.region,
                "population": plan.population,
                "style": style,
                "mean_score": float(scores.mean()),
                "max_score": float(scores.max()),
                "top5_mean": float(np.sort(scores)[-5:].mean()),
            })

        formal = [r for r in results if r["style"] == "formal"]
        informal = [r for r in results if r["style"] == "informal"]

        formal_avg = np.mean([r["top5_mean"] for r in formal]) if formal else 0
        informal_avg = np.mean([r["top5_mean"] for r in informal]) if informal else 0

        return {
            "per_plan": results,
            "formal_avg_top5": float(formal_avg),
            "informal_avg_top5": float(informal_avg),
            "style_gap": float(formal_avg - informal_avg),
            "bias_detected": abs(formal_avg - informal_avg) > 0.05,
        }

    def run_size_bias_test(self) -> dict:
        """
        Test: do large cities get better matches than small towns?
        Could happen if large cities write more detailed plans.
        """
        plans = load_plans()
        grants = load_grants()

        grant_vecs = self.embedder.embed_texts(
            [clean_text(g.description) for g in grants]
        )

        results = []
        for plan in plans:
            plan_vec = self.embedder.embed_long_text(plan.full_text)
            scores = np.dot(grant_vecs, plan_vec)

            results.append({
                "municipality": plan.municipality,
                "population": plan.population,
                "top5_mean": float(np.sort(scores)[-5:].mean()),
                "text_length": len(plan.full_text.split()),
            })

        # correlation between population and score
        pops = np.array([r["population"] for r in results])
        scores = np.array([r["top5_mean"] for r in results])

        # simple correlation
        if len(pops) > 2:
            correlation = float(np.corrcoef(pops, scores)[0, 1])
        else:
            correlation = 0.0

        return {
            "per_plan": results,
            "population_score_correlation": correlation,
            "bias_detected": abs(correlation) > 0.7,
        }

    def run_geographic_bias_test(self) -> dict:
        """
        Test: north vs south Italy — do southern municipalities
        get systematically lower scores?
        """
        south_regions = {"Campania", "Basilicata", "Calabria", "Puglia", "Sicilia", "Sardegna"}

        plans = load_plans()
        grants = load_grants()

        grant_vecs = self.embedder.embed_texts(
            [clean_text(g.description) for g in grants]
        )

        north_scores = []
        south_scores = []

        for plan in plans:
            plan_vec = self.embedder.embed_long_text(plan.full_text)
            scores = np.dot(grant_vecs, plan_vec)
            top5 = float(np.sort(scores)[-5:].mean())

            if plan.region in south_regions:
                south_scores.append(top5)
            else:
                north_scores.append(top5)

        north_avg = float(np.mean(north_scores)) if north_scores else 0
        south_avg = float(np.mean(south_scores)) if south_scores else 0

        return {
            "north_avg_top5": north_avg,
            "south_avg_top5": south_avg,
            "geographic_gap": float(north_avg - south_avg),
            "bias_detected": abs(north_avg - south_avg) > 0.05,
        }

    def full_report(self) -> dict:
        style = self.run_style_bias_test()
        size = self.run_size_bias_test()
        geo = self.run_geographic_bias_test()

        return {"style_bias": style, "size_bias": size, "geographic_bias": geo}


def _detect_style(text: str) -> str:
    """Rough heuristic: informal plans use casual language."""
    informal_markers = [
        "cari concittadini", "cosa vogliamo", "il nostro problema",
        "dobbiamo", "vogliamo", "serve un", "non hanno niente",
        "cascano a pezzi", "che non regge",
    ]
    text_lower = text.lower()
    hits = sum(1 for m in informal_markers if m in text_lower)
    return "informal" if hits >= 2 else "formal"


def format_bias_report(report: dict) -> str:
    lines = ["BIAS DETECTION REPORT\n"]

    # style
    s = report["style_bias"]
    flag = "⚠️  YES" if s["bias_detected"] else "✓  No"
    lines.append(f"1. Writing style bias: {flag}")
    lines.append(f"   Formal plans avg top-5 score: {s['formal_avg_top5']:.3f}")
    lines.append(f"   Informal plans avg top-5 score: {s['informal_avg_top5']:.3f}")
    lines.append(f"   Gap: {s['style_gap']:.3f}\n")

    # size
    z = report["size_bias"]
    flag = "⚠️  YES" if z["bias_detected"] else "✓  No"
    lines.append(f"2. City size bias: {flag}")
    lines.append(f"   Population↔score correlation: {z['population_score_correlation']:.3f}\n")

    # geographic
    g = report["geographic_bias"]
    flag = "⚠️  YES" if g["bias_detected"] else "✓  No"
    lines.append(f"3. Geographic bias (north vs south): {flag}")
    lines.append(f"   North avg top-5: {g['north_avg_top5']:.3f}")
    lines.append(f"   South avg top-5: {g['south_avg_top5']:.3f}")
    lines.append(f"   Gap: {g['geographic_gap']:.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    detector = BiasDetector()
    report = detector.full_report()
    print(format_bias_report(report))
