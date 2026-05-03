"""
Data Loader
============
Loads raw grant and municipal plan data from JSON files
and converts them into typed dataclass instances.

Usage:
    from src.data_collection.loader import load_grants, load_plans

    grants = load_grants()
    plans  = load_plans()
"""

import json
from datetime import date
from pathlib import Path

from src.models import Grant, MunicipalPlan


def _parse_date(date_str: str | None) -> date | None:
    """Convert 'YYYY-MM-DD' string to a date object, or None."""
    if date_str is None:
        return None
    return date.fromisoformat(date_str)


def load_grants(path: Path | None = None) -> list[Grant]:
    """
    Load all EU grants from the JSON database.

    Args:
        path: Override path to the JSON file.
              Defaults to data/raw/grants/eu_grants.json

    Returns:
        List of Grant dataclass instances, sorted by deadline.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "raw" / "grants" / "eu_grants.json"

    with open(path, "r", encoding="utf-8") as f:
        raw_grants = json.load(f)

    grants = []
    for entry in raw_grants:
        grant = Grant(
            id=entry["id"],
            name=entry["name"],
            programme=entry["programme"],
            description=entry["description"],
            themes=entry["themes"],
            budget_eur=entry.get("budget_eur"),
            deadline=_parse_date(entry.get("deadline")),
            url=entry.get("url", ""),
            language=entry.get("language", "en"),
            eligible_entities=entry.get("eligible_entities", []),
        )
        grants.append(grant)

    # Sort by deadline (earliest first), with None-deadline grants last
    grants.sort(key=lambda g: g.deadline or date(2099, 12, 31))
    return grants


def load_plans(path: Path | None = None) -> list[MunicipalPlan]:
    """
    Load all municipal plans from the JSON database.

    Args:
        path: Override path to the JSON file.
              Defaults to data/raw/municipal_plans/plans.json

    Returns:
        List of MunicipalPlan dataclass instances.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "raw" / "municipal_plans" / "plans.json"

    with open(path, "r", encoding="utf-8") as f:
        raw_plans = json.load(f)

    plans = []
    for entry in raw_plans:
        plan = MunicipalPlan(
            id=entry["id"],
            municipality=entry["municipality"],
            region=entry["region"],
            full_text=entry["full_text"],
            population=entry.get("population", 0),
            priorities=entry.get("priorities", []),
            themes=entry.get("themes", []),
            language=entry.get("language", "it"),
        )
        plans.append(plan)

    return plans


def get_grant_by_id(grant_id: str, grants: list[Grant] | None = None) -> Grant | None:
    """Look up a single grant by its ID."""
    if grants is None:
        grants = load_grants()
    for grant in grants:
        if grant.id == grant_id:
            return grant
    return None


def get_plan_by_id(plan_id: str, plans: list[MunicipalPlan] | None = None) -> MunicipalPlan | None:
    """Look up a single plan by its ID."""
    if plans is None:
        plans = load_plans()
    for plan in plans:
        if plan.id == plan_id:
            return plan
    return None


# Quick self-test

if __name__ == "__main__":
    grants = load_grants()
    plans = load_plans()

    print(f"Loaded {len(grants)} grants and {len(plans)} municipal plans.\n")

    print("── Grants (first 5) ──")
    for g in grants[:5]:
        print(f"  {g.id}: {g.name} ({g.programme}) — €{g.budget_eur:,} — deadline {g.deadline}")

    print(f"\n── Municipal Plans ──")
    for p in plans:
        text_preview = p.full_text[:80].replace("\n", " ")
        print(f"  {p.municipality} ({p.region}, pop. {p.population:,}): {text_preview}...")
