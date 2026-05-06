"""
Interactive HTML visualization of the knowledge graph using pyvis.
Generates standalone .html files you can open in a browser.
"""

from pathlib import Path

from config.settings import THEMATIC_DOMAINS
from src.knowledge_graph.builder import load_graph, get_plan_coverage
from src.data_collection.loader import load_grants, get_grant_by_id


COLORS = {
    "plan":              "#4A90D9",
    "domain_covered":    "#2ECC71",
    "domain_uncovered":  "#95A5A6",
    "grant_matched":     "#E67E22",
    "grant_other":       "#BDC3C7",
}


def visualize_plan_graph(
    plan_id: str,
    matched_grant_ids: list[str] | None = None,
    output_path: str | None = None,
    show_unmatched: bool = False,
) -> str:
    """
    Render an interactive graph centered on one plan.
    Returns path to the generated HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("pip install pyvis")
        return ""

    graph = load_graph()
    coverage = get_plan_coverage(plan_id, graph)
    matched_grant_ids = matched_grant_ids or []
    grants = load_grants()

    net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3)

    # plan node at the center
    plan_data = graph.nodes[plan_id]
    net.add_node(
        plan_id,
        label=plan_data.get("label", plan_id),
        color=COLORS["plan"], size=35, shape="star",
        title=f"{plan_data.get('label')} — {plan_data.get('region')}",
    )

    # domain nodes
    for domain in THEMATIC_DOMAINS:
        is_covered = domain in coverage["covered"]
        color = COLORS["domain_covered"] if is_covered else COLORS["domain_uncovered"]

        net.add_node(
            domain,
            label=domain.replace("_", " ").title(),
            color=color,
            size=20 if is_covered else 12,
            title=f"{'✓' if is_covered else '✗'} {domain}",
        )
        if is_covered:
            net.add_edge(plan_id, domain, color=COLORS["domain_covered"], width=2)

    # matched grants
    for gid in matched_grant_ids:
        grant = get_grant_by_id(gid, grants)
        if not grant:
            continue

        label = grant.name[:30] + "..." if len(grant.name) > 30 else grant.name
        budget = f"€{grant.budget_eur:,}" if grant.budget_eur else "N/A"

        net.add_node(
            gid, label=label,
            color=COLORS["grant_matched"], size=18, shape="diamond",
            title=f"{grant.name}\n{grant.programme}\n{budget}",
        )
        for theme in grant.themes:
            if theme in coverage["covered"]:
                net.add_edge(gid, theme, color=COLORS["grant_matched"], width=1)

    # optional: unmatched grants (dimmed)
    if show_unmatched:
        for grant in grants:
            if grant.id in matched_grant_ids:
                continue
            if not any(t in THEMATIC_DOMAINS for t in grant.themes):
                continue

            net.add_node(
                grant.id,
                label=grant.name[:20] + "...",
                color=COLORS["grant_other"], size=10, shape="diamond",
            )
            for theme in grant.themes:
                if theme in THEMATIC_DOMAINS:
                    net.add_edge(grant.id, theme, color="#444444", width=0.5)

    output_path = output_path or f"data/knowledge_graph/{plan_id}_graph.html"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(output_path)
    print(f"  → {output_path}")

    return output_path


if __name__ == "__main__":
    # demo: visualize each plan with a few example grants
    demo_grants = ["LIFE-2026-CET-SOLAR", "ERDF-2026-IT-URBAN", "ESF-2026-YOUTH-EMPLOY"]

    from src.data_collection.loader import load_plans
    for plan in load_plans():
        print(f"{plan.municipality}:")
        visualize_plan_graph(plan.id, matched_grant_ids=demo_grants)
