"""
RAG-based natural language explanations.

Generates human-readable text explaining why a grant matches a plan,
grounded in actual content from both documents.

For now uses a template-based approach. Can be swapped for LLM
generation (OpenAI, HuggingFace, etc.) later.
"""

from src.models import MatchResult
from src.data_collection.loader import get_grant_by_id, get_plan_by_id
from src.knowledge_graph.builder import get_shared_domains
from src.preprocessing.text_cleaner import clean_text


class RagExplainer:
    """
    Three explanation levels:
        - citizen: plain language, no jargon
        - analyst: technical with feature references
        - auditor: full decision trace
    """

    def explain(self, match: MatchResult, level: str = "citizen") -> str:
        grant = get_grant_by_id(match.grant_id)
        plan = get_plan_by_id(match.plan_id)

        if not grant or not plan:
            return "Could not load grant or plan data."

        shared = get_shared_domains(match.plan_id, match.grant_id)
        shared_labels = [d.replace("_", " ") for d in shared]

        if level == "citizen":
            return self._citizen_explanation(plan, grant, match, shared_labels)
        elif level == "analyst":
            return self._analyst_explanation(plan, grant, match, shared_labels)
        elif level == "auditor":
            return self._auditor_explanation(plan, grant, match, shared_labels)
        else:
            return f"Unknown explanation level: {level}"

    def _citizen_explanation(self, plan, grant, match, shared_labels) -> str:
        """Simple narrative a mayor can understand."""

        budget = f"€{grant.budget_eur:,}" if grant.budget_eur else "budget not specified"
        deadline = str(grant.deadline) if grant.deadline else "no deadline listed"

        if not shared_labels:
            theme_text = "general municipal development"
        elif len(shared_labels) == 1:
            theme_text = shared_labels[0]
        else:
            theme_text = ", ".join(shared_labels[:-1]) + f" and {shared_labels[-1]}"

        return (
            f"The {grant.programme} programme has a call called \"{grant.name}\" "
            f"that is relevant to {plan.municipality}'s development plan.\n\n"
            f"Both your plan and this grant focus on {theme_text}. "
            f"The total available budget is {budget}, "
            f"and the application deadline is {deadline}.\n\n"
            f"We recommend reviewing this opportunity as it aligns with "
            f"your municipality's stated priorities."
        )

    def _analyst_explanation(self, plan, grant, match, shared_labels) -> str:
        """Technical explanation with scores and domain mapping."""

        domains_str = ", ".join(shared_labels) if shared_labels else "no direct domain overlap"

        # find relevant snippets from plan text
        plan_snippet = self._extract_relevant_snippet(plan.full_text, grant.themes)
        grant_snippet = clean_text(grant.description)[:300]

        return (
            f"Match: {plan.municipality} ↔ {grant.name}\n"
            f"Score: {match.similarity_score:.4f} | Confidence: {match.confidence}\n"
            f"Shared domains: {domains_str}\n\n"
            f"Plan excerpt (relevant section):\n  \"{plan_snippet}\"\n\n"
            f"Grant description (summary):\n  \"{grant_snippet}...\"\n\n"
            f"The semantic similarity is driven primarily by overlapping themes "
            f"in {domains_str}. See SHAP/LIME analysis for word-level attribution."
        )

    def _auditor_explanation(self, plan, grant, match, shared_labels) -> str:
        """Full trace for audit purposes."""

        return (
            f"=== AUDIT LOG ===\n"
            f"Plan ID:     {match.plan_id}\n"
            f"Grant ID:    {match.grant_id}\n"
            f"Grant Name:  {grant.name}\n"
            f"Programme:   {grant.programme}\n"
            f"Score:       {match.similarity_score:.6f}\n"
            f"Rank:        {match.rank}\n"
            f"Confidence:  {match.confidence}\n"
            f"Shared domains: {', '.join(shared_labels) or 'none'}\n"
            f"All grant themes: {', '.join(grant.themes)}\n"
            f"All plan themes:  {', '.join(get_plan_by_id(match.plan_id).themes)}\n"
            f"Eligible entities: {', '.join(grant.eligible_entities)}\n"
            f"Budget:      €{grant.budget_eur:,}" if grant.budget_eur else "N/A" + "\n"
            f"Deadline:    {grant.deadline}\n"
            f"Grant lang:  {grant.language}\n"
            f"Plan lang:   {get_plan_by_id(match.plan_id).language}\n"
            f"=== END LOG ==="
        )

    def _extract_relevant_snippet(self, text: str, themes: list[str], max_len: int = 200) -> str:
        """
        Find the most relevant paragraph in the plan text
        based on keyword overlap with grant themes.
        Crude but works for template-based explanations.
        """
        theme_words = set()
        for t in themes:
            theme_words.update(t.split("_"))

        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

        if not paragraphs:
            return text[:max_len]

        best_para = paragraphs[0]
        best_overlap = 0

        for para in paragraphs:
            para_lower = para.lower()
            overlap = sum(1 for w in theme_words if w in para_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_para = para

        if len(best_para) > max_len:
            best_para = best_para[:max_len] + "..."

        return best_para


if __name__ == "__main__":
    from src.retrieval.embedder import Embedder
    from src.retrieval.matcher import Matcher

    matcher = Matcher()
    results = matcher.match_plan("plan-frascati", top_k=3)
    explainer = RagExplainer()

    for r in results:
        print(f"\n{'='*60}")
        for level in ["citizen", "analyst", "auditor"]:
            print(f"\n--- {level.upper()} ---")
            print(explainer.explain(r, level=level))
