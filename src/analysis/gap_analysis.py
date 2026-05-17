"""
Gap analysis: what's MISSING from the plan that could unlock more funding?
Inverts the matching logic; instead of "what fits", asks "what doesn't fit yet".
"""

from src.models import FundingGap
from src.knowledge_graph.builder import load_graph, get_plan_coverage, get_domain_grants
from src.data_collection.loader import load_grants, get_grant_by_id


def analyze_gaps(plan_id: str) -> list[FundingGap]:
    """
    For each domain NOT covered by the plan, find grants that fund it
    and estimate the missed budget opportunity.
    """
    graph = load_graph()
    coverage = get_plan_coverage(plan_id, graph)
    grants = load_grants()

    gaps = []
    for domain in coverage["uncovered"]:
        grant_ids = get_domain_grants(domain, graph)
        if not grant_ids:
            continue

        total_budget = 0
        for gid in grant_ids:
            g = get_grant_by_id(gid, grants)
            if g and g.budget_eur:
                total_budget += g.budget_eur

        label = domain.replace("_", " ")
        gaps.append(FundingGap(
            domain=domain,
            potential_grants=grant_ids,
            estimated_budget=total_budget,
            recommendation=(
                f"Your plan does not address {label}. "
                f"Adding initiatives in this area could give access to "
                f"{len(grant_ids)} grant(s) worth up to €{total_budget:,}."
            ),
        ))

    gaps.sort(key=lambda g: g.estimated_budget, reverse=True)
    return gaps


def format_gaps(gaps: list[FundingGap], max_show: int = 8) -> str:
    if not gaps:
        return "No significant gaps found — plan has broad coverage."

    lines = ["Gap analysis — uncovered domains with funding potential:\n"]
    for gap in gaps[:max_show]:
        label = gap.domain.replace("_", " ").title()
        lines.append(
            f"  {label}\n"
            f"    {len(gap.potential_grants)} grant(s), up to €{gap.estimated_budget:,}\n"
            f"    → {gap.recommendation}\n"
        )

    total_missed = sum(g.estimated_budget for g in gaps)
    lines.append(f"  Total missed funding potential: €{total_missed:,}")
    return "\n".join(lines)


if __name__ == "__main__":
    from src.data_collection.loader import load_plans

    for plan in load_plans():
        print(f"\n{'='*60}")
        print(f"  {plan.municipality.upper()}")
        print(f"{'='*60}")
        gaps = analyze_gaps(plan.id)
        print(format_gaps(gaps))
