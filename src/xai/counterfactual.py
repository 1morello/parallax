"""
Counterfactual explanations: "what would need to change?"

Instead of explaining why a match exists, this shows what
modifications to the plan would increase or decrease the match.
"""

import numpy as np

from config.settings import THEMATIC_DOMAINS
from src.models import Counterfactual
from src.retrieval.embedder import Embedder
from src.preprocessing.text_cleaner import clean_text
from src.data_collection.loader import get_grant_by_id


# phrases that represent each domain — used for synthetic additions
DOMAIN_PHRASES = {
    "renewable_energy": "installazione di impianti fotovoltaici e energia solare rinnovabile",
    "energy_efficiency": "riqualificazione energetica degli edifici e riduzione dei consumi",
    "sustainable_transport": "mobilità sostenibile, piste ciclabili e trasporto elettrico",
    "digital_infrastructure": "infrastruttura digitale, banda larga e fibra ottica",
    "digital_skills": "competenze digitali, formazione informatica per cittadini",
    "education": "istruzione, programmi educativi e formazione scolastica",
    "healthcare": "servizi sanitari, telemedicina e assistenza domiciliare",
    "social_inclusion": "inclusione sociale, servizi per famiglie vulnerabili",
    "cultural_heritage": "patrimonio culturale, restauro monumenti e valorizzazione storica",
    "urban_regeneration": "rigenerazione urbana, riqualificazione spazi pubblici",
    "waste_management": "gestione rifiuti, raccolta differenziata e riciclo",
    "water_management": "gestione risorse idriche e infrastrutture fognarie",
    "biodiversity": "biodiversità, aree verdi urbane e corridoi ecologici",
    "climate_adaptation": "adattamento climatico, mitigazione isole di calore",
    "agriculture": "agricoltura sostenibile, filiera corta e prodotti biologici",
    "tourism": "turismo sostenibile, percorsi culturali e promozione territoriale",
    "sme_support": "supporto alle piccole e medie imprese locali",
    "research_innovation": "ricerca e innovazione, laboratori tecnologici",
    "public_administration": "digitalizzazione della pubblica amministrazione",
    "youth_employment": "occupazione giovanile, tirocini e formazione lavoro",
    "gender_equality": "parità di genere, imprenditoria femminile",
    "circular_economy": "economia circolare, riuso e riduzione sprechi",
    "cybersecurity": "sicurezza informatica e protezione dati",
    "housing": "edilizia residenziale pubblica e social housing",
}


class CounterfactualExplainer:

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()

    def explain_additions(self, plan_text: str, grant_id: str,
                          domains_to_try: list[str] | None = None) -> list[Counterfactual]:
        """
        What if we ADD content about new domains to the plan?
        Shows which additions would most increase the match score.
        """
        grant = get_grant_by_id(grant_id)
        grant_vec = self.embedder.embed_text(clean_text(grant.description))

        original_vec = self.embedder.embed_long_text(plan_text)
        original_score = float(np.dot(original_vec, grant_vec))

        domains_to_try = domains_to_try or list(DOMAIN_PHRASES.keys())
        results = []

        for domain in domains_to_try:
            phrase = DOMAIN_PHRASES.get(domain, "")
            if not phrase:
                continue

            modified_text = plan_text + "\n\n" + phrase
            modified_vec = self.embedder.embed_long_text(modified_text)
            modified_score = float(np.dot(modified_vec, grant_vec))

            results.append(Counterfactual(
                grant_id=grant_id,
                original_score=original_score,
                modified_score=modified_score,
                modification=f"Add section on: {domain}",
            ))

        results.sort(key=lambda x: x.delta, reverse=True)
        return results

    def explain_removals(self, plan_text: str, grant_id: str,
                         section_keywords: list[str] | None = None) -> list[Counterfactual]:
        """
        What if we REMOVE parts of the plan?
        Shows which removals would most decrease the match score.

        Uses keyword-based paragraph removal — rough but effective.
        """
        grant = get_grant_by_id(grant_id)
        grant_vec = self.embedder.embed_text(clean_text(grant.description))

        original_vec = self.embedder.embed_long_text(plan_text)
        original_score = float(np.dot(original_vec, grant_vec))

        # split into paragraphs
        paragraphs = [p.strip() for p in plan_text.split("\n\n") if p.strip()]

        if len(paragraphs) < 2:
            # too short to meaningfully remove sections
            return []

        results = []
        for i, paragraph in enumerate(paragraphs):
            # skip very short paragraphs (headers, etc)
            if len(paragraph.split()) < 10:
                continue

            remaining = "\n\n".join(p for j, p in enumerate(paragraphs) if j != i)
            modified_vec = self.embedder.embed_long_text(remaining)
            modified_score = float(np.dot(modified_vec, grant_vec))

            # first ~60 chars as label
            label = paragraph[:60].replace("\n", " ") + "..."

            results.append(Counterfactual(
                grant_id=grant_id,
                original_score=original_score,
                modified_score=modified_score,
                modification=f"Remove: \"{label}\"",
            ))

        results.sort(key=lambda x: x.delta, reverse=True)
        return results

    def format_explanation(self, counterfactuals: list[Counterfactual], mode: str = "additions") -> str:
        lines = [f"Counterfactual analysis ({mode}):\n"]

        for cf in counterfactuals[:8]:
            arrow = "↑" if cf.direction == "increase" else "↓"
            lines.append(
                f"  {arrow} {cf.delta:+.4f}  ({cf.original_score:.3f} → {cf.modified_score:.3f})"
                f"  {cf.modification}"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    from src.data_collection.loader import load_plans

    explainer = CounterfactualExplainer()
    plan = load_plans()[0]  # Frascati

    print("=== What if we ADD new topics? ===\n")
    additions = explainer.explain_additions(plan.full_text, "ERDF-2026-IT-URBAN")
    print(explainer.format_explanation(additions, "additions"))

    print("\n\n=== What if we REMOVE sections? ===\n")
    removals = explainer.explain_removals(plan.full_text, "ERDF-2026-IT-URBAN")
    print(explainer.format_explanation(removals, "removals"))
