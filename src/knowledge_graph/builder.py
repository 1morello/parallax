"""
Builds a tripartite graph: Plans - Domains - Grants.
Enables structured explanations and gap analysis.
"""

import pickle

import networkx as nx

from config.settings import THEMATIC_DOMAINS, KNOWLEDGE_GRAPH_DIR, GRAPH_PKL_PATH
from src.data_collection.loader import load_grants, load_plans


def build_graph() -> nx.Graph:
    """
    Construct the full knowledge graph.

    Node types: "plan", "domain", "grant"
    Edge types: "covers" (plan→domain), "funds" (grant→domain)
    """
    G = nx.Graph()

    for domain in THEMATIC_DOMAINS:
        G.add_node(domain, node_type="domain", label=_fmt(domain))

    for plan in load_plans():
        G.add_node(plan.id, node_type="plan", label=plan.municipality,
                   region=plan.region, population=plan.population)
        for theme in plan.themes:
            if theme in THEMATIC_DOMAINS:
                G.add_edge(plan.id, theme, edge_type="covers")

    for grant in load_grants():
        G.add_node(grant.id, node_type="grant", label=grant.name,
                   programme=grant.programme, budget=grant.budget_eur,
                   deadline=str(grant.deadline) if grant.deadline else None)
        for theme in grant.themes:
            if theme in THEMATIC_DOMAINS:
                G.add_edge(grant.id, theme, edge_type="funds")

    KNOWLEDGE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PKL_PATH, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {GRAPH_PKL_PATH}")
    return G


def load_graph() -> nx.Graph:
    with open(GRAPH_PKL_PATH, "rb") as f:
        return pickle.load(f)


def get_plan_coverage(plan_id: str, graph: nx.Graph | None = None) -> dict:
    """Which domains does this plan cover vs not cover?"""
    graph = graph or load_graph()

    covered = [n for n in graph.neighbors(plan_id)
               if graph.nodes[n].get("node_type") == "domain"]

    uncovered = [d for d in THEMATIC_DOMAINS if d not in covered]

    return {
        "covered": sorted(covered),
        "uncovered": sorted(uncovered),
        "coverage_ratio": len(covered) / len(THEMATIC_DOMAINS),
    }


def get_domain_grants(domain: str, graph: nx.Graph | None = None) -> list[str]:
    """All grant IDs that fund a given domain."""
    graph = graph or load_graph()
    return [n for n in graph.neighbors(domain)
            if graph.nodes[n].get("node_type") == "grant"]


def get_shared_domains(plan_id: str, grant_id: str, graph: nx.Graph | None = None) -> list[str]:
    """Domains that both the plan and grant connect to."""
    graph = graph or load_graph()

    plan_domains = {n for n in graph.neighbors(plan_id)
                    if graph.nodes[n].get("node_type") == "domain"}
    grant_domains = {n for n in graph.neighbors(grant_id)
                     if graph.nodes[n].get("node_type") == "domain"}

    return sorted(plan_domains & grant_domains)


def _fmt(domain: str) -> str:
    return domain.replace("_", " ").title()


if __name__ == "__main__":
    graph = build_graph()
    print()

    for plan in load_plans():
        cov = get_plan_coverage(plan.id, graph)
        n = len(cov["covered"])
        total = len(THEMATIC_DOMAINS)
        print(f"{plan.municipality}: {n}/{total} domains ({cov['coverage_ratio']:.0%})")
        print(f"  covered:   {', '.join(cov['covered'][:6])}{'...' if n > 6 else ''}")
        print(f"  uncovered: {', '.join(cov['uncovered'][:6])}{'...' if len(cov['uncovered']) > 6 else ''}")
        print()
