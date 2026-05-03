"""
PARALLAX Configuration
======================
Central place for all paths, constants, and model settings.
Import this module anywhere to get consistent configuration.
"""

from pathlib import Path


# ── Project Root ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"


# ── Data Paths ────────────────────────────────────────────

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

GRANTS_RAW_DIR = RAW_DATA_DIR / "grants"
PLANS_RAW_DIR = RAW_DATA_DIR / "municipal_plans"

GRANTS_DB_PATH = PROCESSED_DATA_DIR / "grants_embedded.pkl"
PLANS_DB_PATH = PROCESSED_DATA_DIR / "plans_embedded.pkl"

KNOWLEDGE_GRAPH_DIR = DATA_DIR / "knowledge_graph"
GRAPH_NODES_PATH = KNOWLEDGE_GRAPH_DIR / "domains.json"
GRAPH_EDGES_PATH = KNOWLEDGE_GRAPH_DIR / "edges.json"
GRAPH_PKL_PATH = KNOWLEDGE_GRAPH_DIR / "graph.pkl"


# ── Model Settings ────────────────────────────────────────

# Multilingual sentence-transformer for cross-lingual matching
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# How many top grants to return per query
TOP_K_RESULTS = 10

# Minimum similarity score to consider a match relevant
SIMILARITY_THRESHOLD = 0.35


# ── Knowledge Graph Domains ───────────────────────────────
# These are the thematic domains that bridge municipal plans
# and EU grants. Each plan and grant gets mapped to one or
# more of these domains.

THEMATIC_DOMAINS = [
    "renewable_energy",
    "energy_efficiency",
    "sustainable_transport",
    "digital_infrastructure",
    "digital_skills",
    "education",
    "healthcare",
    "social_inclusion",
    "cultural_heritage",
    "urban_regeneration",
    "waste_management",
    "water_management",
    "biodiversity",
    "climate_adaptation",
    "agriculture",
    "tourism",
    "sme_support",
    "research_innovation",
    "public_administration",
    "youth_employment",
    "gender_equality",
    "circular_economy",
    "cybersecurity",
    "housing",
]


# ── XAI Settings ──────────────────────────────────────────

# Number of perturbations for LIME explanations
LIME_NUM_SAMPLES = 500

# Number of features to show in SHAP/LIME plots
XAI_TOP_FEATURES = 15

# Counterfactual: how many tokens to perturb at a time
COUNTERFACTUAL_PERTURBATION_RATE = 0.15


# ── Confidence Calibration ────────────────────────────────

# If the gap between rank-1 and rank-2 scores is below this,
# we flag the result as "low confidence"
CONFIDENCE_GAP_THRESHOLD = 0.08

# If more than this many results are within this range of
# the top score, we flag as "ambiguous"
CONFIDENCE_CLUSTER_RANGE = 0.05
CONFIDENCE_CLUSTER_MAX_SIZE = 5


# ── RAG Settings ──────────────────────────────────────────

RAG_CHUNK_SIZE = 512        # tokens per chunk
RAG_CHUNK_OVERLAP = 64      # overlap between chunks
RAG_CONTEXT_WINDOW = 3      # number of chunks to feed to LLM


# ── Explanation Levels ────────────────────────────────────

EXPLANATION_LEVELS = {
    "citizen": {
        "label": "For the Mayor / Citizen",
        "description": "Simple natural language explanation",
        "format": "narrative",
    },
    "analyst": {
        "label": "For the Analyst",
        "description": "Feature importance with SHAP/LIME visuals",
        "format": "technical",
    },
    "auditor": {
        "label": "For the Auditor",
        "description": "Full decision trace with confidence scores",
        "format": "audit_log",
    },
}
