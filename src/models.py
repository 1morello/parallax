"""
PARALLAX Data Models
================
Dataclass definitions for the core entities in the system:
grants, municipal plans, matches, and explanations.

These models enforce a consistent structure across all modules
and make the code self-documenting.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ── EU Grant ──────────────────────────────────────────────

@dataclass
class Grant:
    """
    Represents a single EU funding programme or call.

    Attributes:
        id:           Unique identifier (e.g., "LIFE-2026-CET-01")
        name:         Human-readable title
        programme:    Parent programme (e.g., "Horizon Europe", "LIFE")
        description:  Full text description of the call
        themes:       List of thematic domain tags (from THEMATIC_DOMAINS)
        budget_eur:   Total available budget in euros (None if unspecified)
        deadline:     Application deadline
        url:          Link to official call page
        language:     Language of the description ("en", "it", or "multi")
        eligible_entities: Who can apply (e.g., ["municipalities", "SMEs"])
    """
    id: str
    name: str
    programme: str
    description: str
    themes: list[str]
    budget_eur: Optional[int] = None
    deadline: Optional[date] = None
    url: str = ""
    language: str = "en"
    eligible_entities: list[str] = field(default_factory=list)


# ── Municipal Plan ────────────────────────────────────────

@dataclass
class MunicipalPlan:
    """
    Represents a municipal development plan or strategic document.

    Attributes:
        id:             Unique identifier
        municipality:   Name of the municipality
        region:         Italian region (e.g., "Lazio", "Campania")
        population:     Population count (for bias analysis)
        full_text:      Complete text of the plan
        priorities:     Extracted key priorities (after summarization)
        themes:         Mapped thematic domains
        language:       Language of the document ("it" or "en")
    """
    id: str
    municipality: str
    region: str
    full_text: str
    population: int = 0
    priorities: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    language: str = "it"


# ── Match Result ──────────────────────────────────────────

@dataclass
class MatchResult:
    """
    A single matching between a municipal plan and a grant.

    Attributes:
        plan_id:            Which plan this match is for
        grant_id:           Which grant was matched
        similarity_score:   Cosine similarity (0.0 to 1.0)
        rank:               Position in the ranked list (1 = best)
        confidence:         Calibrated confidence level
        shared_themes:      Overlapping thematic domains
        explanation:        Dict with keys "citizen", "analyst", "auditor"
    """
    plan_id: str
    grant_id: str
    similarity_score: float
    rank: int
    confidence: str = "medium"        # "high", "medium", "low"
    shared_themes: list[str] = field(default_factory=list)
    explanation: dict = field(default_factory=dict)


# ── Gap ───────────────────────────────────────────────────

@dataclass
class FundingGap:
    """
    Identifies a thematic area NOT covered by the municipal plan
    that could unlock additional funding.

    Attributes:
        domain:           The uncovered thematic domain
        potential_grants:  List of grant IDs that would become relevant
        estimated_budget:  Sum of budgets from those grants
        recommendation:   Plain-language suggestion for the municipality
    """
    domain: str
    potential_grants: list[str] = field(default_factory=list)
    estimated_budget: int = 0
    recommendation: str = ""


# ── Counterfactual ────────────────────────────────────────

@dataclass
class Counterfactual:
    """
    A 'what-if' explanation showing how changes to the plan
    would affect matching results.

    Attributes:
        grant_id:          The grant being explained
        original_score:    Score with the original plan text
        modified_score:    Score after the modification
        modification:      Description of what was changed
        direction:         "increase" or "decrease"
        delta:             Absolute change in score
    """
    grant_id: str
    original_score: float
    modified_score: float
    modification: str
    direction: str = ""
    delta: float = 0.0

    def __post_init__(self):
        self.delta = abs(self.modified_score - self.original_score)
        self.direction = (
            "increase" if self.modified_score > self.original_score
            else "decrease"
        )
