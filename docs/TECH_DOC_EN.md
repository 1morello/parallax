# PARALLAX — Technical Documentation

> A shift in perspective reveals what was always there.

---

## 1. Project Overview

### 1.1 What are we building?

PARALLAX is an AI system that helps European municipalities find EU funding opportunities that match their development plans. It does three things:

1. **Match** — takes a municipal plan document and finds the most relevant EU grants using semantic search
2. **Explain** — provides multi-level explanations of *why* each grant matches, tailored to different audiences
3. **Advise** — identifies gaps in the plan (themes not covered) that could unlock additional funding

### 1.2 Why does this matter?

The EU allocates billions of euros annually through hundreds of grant programmes. Small municipalities don't have the staff or expertise to navigate this landscape. Money goes unspent. PARALLAX bridges this gap.

### 1.3 The EY Challenge

This project is part of EY's AI Explainability challenge for the LUISS AI Techniques course. EY's brief focuses on **XAI (Explainable AI)** — ensuring that AI decisions are transparent, trustworthy, and auditable. The core question: *can we not only find the right grant, but also explain why it's right, in a way that different stakeholders can understand and trust?*

### 1.4 Evaluation Criteria (from EY)

| Criterion | Weight | Question |
|-----------|--------|----------|
| Innovation | 10 pts | Is there a differentiating element? |
| Technical Implementation | 5 pts | Is the system manageable? |
| Accountability & Design | 5 pts | Does the model work correctly? |
| XAI Quality | 5 pts | Is the explanation understandable? |
| Communication | 5 pts | Is the value proposition clearly presented? |

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
INPUT                    PROCESSING                      OUTPUT
─────                    ──────────                      ──────

Municipal Plan ──┐
                 │
                 ▼
          ┌─────────────┐
          │ Preprocessor │──── chunk, clean, summarize
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐     ┌───────────────┐
          │  Embedder    │────▶│ Grant Vectors  │
          │ (sentence-   │     │ (pre-computed) │
          │  transformer)│     └───────────────┘
          └──────┬──────┘
                 │ cosine similarity
                 ▼
          ┌─────────────┐
          │   Matcher    │──── ranked list of grants
          └──────┬──────┘
                 │
        ┌────────┼────────┬──────────────┐
        ▼        ▼        ▼              ▼
   ┌────────┐ ┌──────┐ ┌──────────┐ ┌────────────┐
   │  SHAP  │ │LIME  │ │Counter-  │ │    RAG     │
   │        │ │      │ │factual   │ │ Narrative  │
   └───┬────┘ └──┬───┘ └────┬─────┘ └─────┬──────┘
       └─────────┴──────────┴──────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Knowledge Graph  │──── domain mapping
              └────────┬──────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   Gap    │ │  Bias    │ │Confidence│
    │ Analysis │ │Detection │ │Calibrate │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         └────────────┴────────────┘
                      │
                      ▼
            ┌────────────────┐
            │   Streamlit    │
            │   Dashboard    │
            └────────────────┘
```

### 2.2 Module Responsibilities

| Module | Directory | What it does |
|--------|-----------|--------------|
| Data Models | `src/models.py` | Defines Grant, MunicipalPlan, MatchResult, FundingGap, Counterfactual |
| Data Loader | `src/data_collection/` | Loads grants and plans from JSON into typed objects |
| Preprocessor | `src/preprocessing/` | Cleans text, splits into chunks, handles multilingual content |
| Retrieval | `src/retrieval/` | Embeds text with sentence-transformers, computes similarity, ranks |
| Knowledge Graph | `src/knowledge_graph/` | Builds domain graph (plan ↔ themes ↔ grants), enables structured explanations |
| XAI | `src/xai/` | SHAP, LIME, counterfactual, RAG narrative generation |
| Analysis | `src/analysis/` | Gap analysis, bias detection, confidence calibration |
| Dashboard | `src/app/` | Streamlit interactive UI with What-If explorer |
| Config | `config/` | All constants, paths, model settings, thematic domains |

---

## 3. Data

### 3.1 EU Grants Database

Located at `data/raw/grants/eu_grants.json`.

We have **29 grants** from real EU programmes. Each grant has:

```python
{
    "id": "LIFE-2026-CET-BUILDREN",       # unique identifier
    "name": "Building Renovation...",       # human-readable title
    "programme": "LIFE Clean Energy...",    # parent EU programme
    "description": "This call supports...",  # full text (the main content for matching)
    "themes": ["energy_efficiency", ...],    # tagged thematic domains
    "budget_eur": 18000000,                  # total budget in euros
    "deadline": "2026-09-15",                # application deadline
    "url": "https://...",                    # link to official page
    "language": "en",                        # "en" or "it"
    "eligible_entities": ["municipalities"]   # who can apply
}
```

**Programmes represented:** LIFE, Horizon Europe, Erasmus+, ERDF, Digital Europe, CEF, ESF+, Interreg, CERV, Creative Europe, PNRR, AMIF, Just Transition Fund, COSME, Agricultural Promotion.

**Languages:** 24 grants in English, 5 in Italian (for cross-lingual testing).

### 3.2 Municipal Plans

Located at `data/raw/municipal_plans/plans.json`.

We have **5 plans** chosen for maximum diversity:

| Municipality | Region | Population | Language | Style | Purpose |
|---|---|---|---|---|---|
| Frascati | Lazio | 22,000 | IT | Formal | Baseline medium city |
| Montescaglioso | Basilicata | 9,200 | IT | Informal/colloquial | Bias testing (southern, casual writing) |
| Bolzano | Trentino | 108,000 | EN | Formal/technical | Cross-lingual test, large city |
| Casal Velino | Campania | 5,100 | IT | Informal | Small southern town, bias testing |
| Modena | Emilia-Romagna | 187,000 | EN | Formal/technical | Innovation-heavy, large city |

This diversity is **intentional** — it enables:
- Cross-lingual matching (IT plans ↔ EN grants)
- Bias detection (formal vs informal writing style)
- Scale testing (5K population vs 187K)
- Geographic diversity (north vs south Italy)

### 3.3 Thematic Domains

Defined in `config/settings.py`. There are **24 domains** that serve as the "bridge" vocabulary between plans and grants:

`renewable_energy`, `energy_efficiency`, `sustainable_transport`, `digital_infrastructure`, `digital_skills`, `education`, `healthcare`, `social_inclusion`, `cultural_heritage`, `urban_regeneration`, `waste_management`, `water_management`, `biodiversity`, `climate_adaptation`, `agriculture`, `tourism`, `sme_support`, `research_innovation`, `public_administration`, `youth_employment`, `gender_equality`, `circular_economy`, `cybersecurity`, `housing`

Both grants and plans are tagged with these domains. The Knowledge Graph uses them as intermediate nodes.

---

## 4. Core Technologies

### 4.1 Sentence Transformers (Retrieval)

**What:** Pre-trained neural network that converts any text into a fixed-size vector (embedding) of 384 dimensions.

**Model:** `paraphrase-multilingual-MiniLM-L12-v2`

**Why this model:**
- Multilingual — understands 50+ languages, so Italian plans can match English grants without translation
- Small and fast — 118M parameters, runs on CPU
- Good quality — strong performance on semantic similarity benchmarks
- Available from HuggingFace via the `sentence-transformers` library (which is in the course's software list)

**How it works:**
```
"Installazione pannelli solari sugli edifici comunali"
                    │
                    ▼ sentence-transformer
        [0.023, -0.156, 0.891, ..., 0.034]  ← 384-dimensional vector
```

Two texts about the same topic will have vectors that point in similar directions. We measure this with **cosine similarity** (1.0 = identical meaning, 0.0 = completely unrelated).

### 4.2 SHAP (XAI)

**What:** SHapley Additive exPlanations. Based on game theory (Shapley values from cooperative game theory). Treats each input feature as a "player" and computes how much each player contributed to the final result.

**In our context:** Input features are words/phrases from the municipal plan. SHAP tells us: "The word 'fotovoltaico' contributed +0.12 to the similarity score with grant X, while 'turismo' contributed -0.03."

**Output:** Bar chart showing top positive and negative contributors.

**Library:** `shap`

### 4.3 LIME (XAI)

**What:** Local Interpretable Model-agnostic Explanations. Creates a simple, interpretable model (like linear regression) that approximates the complex model's behavior *locally* — just around one specific prediction.

**How:** Takes the input text, randomly removes words, observes how the similarity score changes, and builds a local linear model from these observations.

**In our context:** For a specific plan-grant match, LIME says: "If you remove 'riqualificazione energetica' from the plan, the match drops from 0.87 to 0.41 — so that phrase is critical."

**Library:** `lime`

**SHAP vs LIME:** SHAP is mathematically grounded and consistent. LIME is faster and more intuitive. We use both — their agreement strengthens trust, their disagreement signals interesting edge cases.

### 4.4 Counterfactual Explanations (XAI)

**What:** Answers "what would need to change for a different outcome?" Instead of explaining the current decision, it shows the *minimal change* needed to alter it.

**In our context:** "If you added a section on 'digital inclusion for elderly citizens' to your plan, this grant's match score would increase from 0.65 to 0.89." Or: "If you removed the 'sustainable transport' section, you would lose your top 3 matches."

**Implementation:** We systematically add/remove thematic content from the plan text, re-run the similarity computation, and measure the delta.

**Why it matters:** This is the most actionable form of explanation. It tells the municipality *what to do*, not just *what happened*.

### 4.5 RAG — Retrieval-Augmented Generation (XAI)

**What:** A technique where we first retrieve relevant documents, then feed them to an LLM to generate a natural-language explanation grounded in actual source material.

**In our context:**
1. Retrieval step finds: "Plan section 2.1 mentions solar panels" + "Grant LIFE-2026-CET-SOLAR funds solar deployment on public buildings"
2. LLM generates: "Your plan's priority to install solar panels on municipal buildings (section 2.1) directly aligns with the LIFE programme's call for solar deployment on public infrastructure. Both focus on reducing energy costs for public facilities."

**Why not just use an LLM directly?** Without RAG, the LLM might hallucinate — inventing connections that don't exist. RAG grounds the explanation in actual document content.

### 4.6 Knowledge Graph

**What:** A graph data structure where nodes represent entities (plans, grants, thematic domains) and edges represent relationships between them.

**Our graph structure:**
```
     [Plan: Frascati] ────covers────▶ [Domain: renewable_energy] ◀────funds──── [Grant: LIFE-SOLAR]
                       ────covers────▶ [Domain: digital_skills]  ◀────funds──── [Grant: DIGITAL-SKILLS]
                                       [Domain: biodiversity]    ◀────funds──── [Grant: LIFE-NAT-BIO]
                                           ▲
                                           │
                                      NOT covered by Frascati → GAP!
```

**Why a graph?** It enables structured explanations ("your plan covers 8 out of 24 domains, here's what's missing") and powers the Gap Analysis module.

**Library:** `networkx` for the graph, `pyvis` for interactive visualization.

### 4.7 Streamlit (Dashboard)

**What:** Python library for building web apps with minimal code. No HTML/CSS/JavaScript needed.

**Our dashboard features:**
- Upload/select a municipal plan
- View ranked grant matches with similarity scores
- Toggle between 3 explanation levels (citizen / analyst / auditor)
- Interactive What-If explorer: add/remove priorities, see how matches change in real-time
- Knowledge graph visualization
- Gap analysis with recommendations
- Bias detection report
- Confidence indicators for each match

---

## 5. Key Concepts Explained

### 5.1 The Three Explanation Levels ("The Parallax")

This is our core differentiator. The same match result is explained differently depending on who's looking:

**Level 1 — Citizen / Mayor:**
Plain text narrative. No jargon, no numbers. Example:
> "Il bando LIFE per l'efficienza energetica è adatto al vostro piano perché entrambi si concentrano sulla riqualificazione delle scuole e degli edifici pubblici per ridurre i consumi energetici."

**Level 2 — Analyst:**
SHAP/LIME visualizations showing which words/phrases drove the match score. Feature importance bar charts. Counterfactual analysis.

**Level 3 — Auditor:**
Full decision log: input data hash, model version, embedding vectors, similarity computation, confidence score, list of all grants considered and why each was ranked where it was. Complete traceability.

### 5.2 Gap Analysis

Gap analysis inverts the matching process. Instead of asking "which grants match my plan?", it asks "which grants *don't* match — and what would I need to add to my plan to access them?"

Implementation:
1. Map the plan to thematic domains (via Knowledge Graph)
2. Find domains NOT covered by the plan
3. Find grants that fund those uncovered domains
4. Generate recommendations: "Adding a section on biodiversity could unlock 3 additional grants worth €47M"

### 5.3 Bias Detection

We test whether the system treats all municipalities fairly, regardless of:
- **Writing style** — formal bureaucratic language vs. casual/conversational
- **City size** — large cities with detailed plans vs. small towns with simple ones
- **Geographic region** — northern vs. southern Italy

**Method:** Take semantically equivalent plans written in different styles, run them through the system, compare results. If Montescaglioso (casual style) systematically gets lower match scores than Bolzano (formal style) for similar content — that's bias.

**Why this matters:** EY's XAI pillars include "Fairness & Non-Discrimination". Most teams will ignore this. We don't.

### 5.4 Confidence Calibration

Not all match scores are equally trustworthy. Confidence calibration adds a meta-layer that tells the user *how much to trust* each result:

- **High confidence:** Top grant scores 0.89, next one scores 0.62. Clear winner.
- **Medium confidence:** Top grant scores 0.75, several others at 0.70-0.73. Ambiguous.
- **Low confidence:** Plan text is very short or vague. Not enough signal to match reliably.

---

## 6. File Structure Reference

```
parallax/
│
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
│
├── config/
│   ├── __init__.py
│   └── settings.py                    # All paths, constants, model params
│
├── data/
│   ├── raw/
│   │   ├── grants/
│   │   │   └── eu_grants.json         # 29 EU grant programmes
│   │   └── municipal_plans/
│   │       └── plans.json             # 5 diverse municipal plans
│   ├── processed/                     # Generated: embeddings, features
│   └── knowledge_graph/               # Generated: graph data
│
├── src/
│   ├── __init__.py
│   ├── models.py                      # Grant, MunicipalPlan, MatchResult, etc.
│   │
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── loader.py                  # Load JSON → dataclass objects
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py            # Clean, normalize text
│   │   └── chunker.py                 # Split long docs into chunks
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py                # sentence-transformer encoding
│   │   └── matcher.py                 # cosine similarity ranking
│   │
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── builder.py                 # Construct graph from data
│   │   └── visualizer.py              # Render interactive graph
│   │
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py          # SHAP-based explanations
│   │   ├── lime_explainer.py          # LIME-based explanations
│   │   ├── counterfactual.py          # What-if analysis
│   │   └── rag_explainer.py           # LLM narrative generation
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── gap_analysis.py            # Missing domain detection
│   │   ├── bias_detection.py          # Fairness testing
│   │   └── confidence.py              # Score calibration
│   │
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py           # Main dashboard
│
├── tests/                             # Unit tests
├── notebooks/                         # Exploration & prototyping
└── docs/
    ├── TECH_DOC_EN.md                 # This file
    └── TECH_DOC_RU.md                 # Russian version
```

---

## 7. Development Phases

### Phase 0: Data Foundation (done ✓)
- [x] Project structure
- [x] Data models
- [x] Grant database (29 entries)
- [x] Municipal plans (5 entries)
- [x] Data loader
- [x] Configuration

### Phase 1: Core Retrieval Engine
- [ ] Text preprocessor (cleaning, chunking)
- [ ] Sentence-transformer embedding
- [ ] Cosine similarity matcher
- [ ] Basic ranked results

### Phase 2: Knowledge Graph
- [ ] Domain taxonomy definition
- [ ] Plan → domain mapping
- [ ] Grant → domain mapping
- [ ] Graph construction (NetworkX)
- [ ] Graph visualization (pyvis)

### Phase 3: XAI Engine
- [ ] SHAP explainer
- [ ] LIME explainer
- [ ] Counterfactual generator
- [ ] RAG narrative explainer
- [ ] Three-level explanation formatter

### Phase 4: Analysis Layer
- [ ] Gap analysis
- [ ] Bias detection
- [ ] Confidence calibration

### Phase 5: Dashboard
- [ ] Streamlit app skeleton
- [ ] Plan upload/selection
- [ ] Match results display
- [ ] Explanation level toggle
- [ ] What-If interactive explorer
- [ ] Knowledge graph view
- [ ] Gap analysis panel

### Phase 6: Documentation & Submission
- [ ] Technical report (5-10 pages PDF)
- [ ] Presentation (max 5 slides)
- [ ] Code cleanup, comments, README in src/
- [ ] requirements.txt verification
- [ ] CRediT statement
- [ ] GenAI disclosure

---

## 8. Key Design Decisions

### Why sentence-transformers and not TF-IDF?
TF-IDF is bag-of-words — it can't understand that "pannelli solari" and "solar photovoltaic systems" mean the same thing. Sentence-transformers capture semantic meaning, which is essential for cross-lingual matching.

### Why a Knowledge Graph and not just embeddings?
Embeddings give you a similarity score. A Knowledge Graph gives you *structure* — which domains are covered, which aren't, how they relate. This powers Gap Analysis and makes explanations more concrete than "these texts are 87% similar."

### Why three XAI methods instead of one?
Each method has strengths: SHAP is rigorous, LIME is intuitive, counterfactuals are actionable, RAG narratives are human-friendly. Using multiple methods and showing their agreement (or disagreement) is a stronger argument for trustworthiness than relying on any single one.

### Why diverse municipal plans?
To test for bias. If we only had formal plans from large northern cities, we'd never know if the system discriminates against small southern towns with informal writing. The diversity is a feature, not a convenience.

---

## 9. References

- Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*. [LIME paper]
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*. [SHAP paper]
- Wachter, Mittelstadt & Russell (2017). Counterfactual Explanations without Opening the Black Box. *Harvard Journal of Law & Technology*. [Counterfactual XAI]
- Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*. [Sentence transformers]
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. [RAG]
- EU AI Act — Regulation (EU) 2024/1689
- EY presentation: "AI Explainability: how to ensure reliability of AI's outcomes" (05.03.2026)
