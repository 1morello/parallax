# parallax

## About

European municipalities have access to billions in EU funding but lack the resources to navigate hundreds of grant programmes. Parallax matches a city's development plan against available grants using semantic search, then explains each match through SHAP/LIME, counterfactual analysis, and LLM-generated narratives — tailored to three audience levels (citizen, analyst, auditor).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Build embeddings and run matching:

```bash
python -m src.preprocessing.pipeline
```

Build knowledge graph:

```bash
python -m src.knowledge_graph.builder
```

Launch the app:

```bash
uvicorn src.app.api:app --reload
```

**Company:** EY — AI Explainability Challenge  
