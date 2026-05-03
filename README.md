# PARALLAX

**An AI system that matches municipal development plans with EU funding opportunities, featuring multi-perspective explainability (XAI), gap analysis, and bias-aware recommendations.**

> *Parallax* — a shift in perspective reveals what was always there.  
> The same AI decision, explained from three viewpoints: citizen, analyst, auditor.

---

## What is this?

EU municipalities have access to billions in funding, but the landscape is fragmented across hundreds of programmes. PARALLAX finds the right grants for a city's development plan — and explains *why* each one fits, from multiple perspectives.

## Architecture

```
Municipal Plan (IT/EN)
        │
        ▼
┌─────────────────────┐
│  Text Preprocessing  │
│  & Summarization     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────┐
│  Multilingual        │────▶│  Grant Database   │
│  Semantic Retrieval  │     │  (embeddings)     │
└────────┬────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────────┐
│  Knowledge Graph     │
│  Enrichment          │
└────────┬────────────┘
         │
    ┌────┴─────┬──────────────┬────────────────┐
    ▼          ▼              ▼                ▼
┌────────┐ ┌────────┐ ┌────────────┐ ┌──────────────┐
│ SHAP/  │ │Counter-│ │    RAG     │ │     Gap      │
│ LIME   │ │factual │ │ Narrative  │ │   Analysis   │
└───┬────┘ └───┬────┘ └─────┬──────┘ └──────┬───────┘
    └──────────┴─────────────┴───────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Streamlit Dashboard   │
         │   (What-If Explorer)    │
         └────────────────────────┘
```

## Project Structure

```
parallax/
├── config/              # Configuration and constants
├── data/
│   ├── raw/             # Original grant & plan data
│   ├── processed/       # Cleaned, embedded data
│   └── knowledge_graph/ # Graph data
├── src/
│   ├── models.py        # Core data models
│   ├── data_collection/ # Data loading
│   ├── preprocessing/   # Text cleaning and chunking
│   ├── retrieval/       # Embedding and semantic matching
│   ├── knowledge_graph/ # Graph construction
│   ├── xai/             # All explainability modules
│   ├── analysis/        # Gap analysis, bias, confidence
│   └── app/             # Streamlit dashboard
├── tests/
├── notebooks/
└── docs/                # Technical documentation
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

## Documentation

- **[Technical Documentation (EN)](docs/TECH_DOC_EN.md)** — full system reference
- **[Техническая документация (RU)](docs/TECH_DOC_RU.md)** — полный справочник

## Team

| Name | Role | Email |
|------|------|-------|
| TBD  | TBD  | TBD   |

**Company:** EY — AI Explainability Challenge  
**Course:** Artificial Intelligence Techniques — LUISS 2025/2026
