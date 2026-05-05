"""
Runs the full Phase 1 pipeline: embed everything, match, print results.

    python -m src.preprocessing.pipeline
"""

import numpy as np

from src.data_collection.loader import load_grants, load_plans
from src.retrieval.embedder import Embedder, build_grant_embeddings, build_plan_embeddings
from src.retrieval.matcher import Matcher, format_results


def main():
    grants = load_grants()
    plans = load_plans()
    print(f"Loaded {len(grants)} grants, {len(plans)} plans.\n")

    emb = Embedder()

    # quick cross-lingual check before we commit to full embedding
    v1 = emb.embed_text("Installazione di pannelli solari sugli edifici")
    v2 = emb.embed_text("Solar panel installation on buildings")
    v3 = emb.embed_text("Medieval fortress archaeological survey")
    print(f"Sanity: IT↔EN solar = {np.dot(v1, v2):.3f}, solar↔medieval = {np.dot(v1, v3):.3f}\n")

    build_grant_embeddings(emb)
    build_plan_embeddings(emb)

    print("\n" + "=" * 60)
    print("MATCHING RESULTS")
    print("=" * 60)

    matcher = Matcher(embedder=emb)
    for plan in plans:
        print(f"\n── {plan.municipality.upper()} ({plan.region}, {plan.population:,}) ──")
        print(format_results(matcher.match_plan(plan.id, top_k=5), grants))

    print("\n✓ Phase 1 done.")


if __name__ == "__main__":
    main()
