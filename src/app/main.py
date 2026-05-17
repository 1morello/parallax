
from nicegui import ui
import numpy as np

from src.data_collection.loader import load_plans, load_grants, get_grant_by_id, get_plan_by_id
from src.retrieval.matcher import Matcher
from src.knowledge_graph.builder import load_graph, get_plan_coverage
from src.analysis.gap_analysis import analyze_gaps
from src.analysis.confidence import analyze_score_distribution
from src.xai.rag_explainer import RagExplainer

# load everything once at startup
plans = load_plans()
grants = load_grants()
matcher = Matcher()
rag = RagExplainer()


def create_header():
    with ui.header().classes("bg-dark"):
        ui.label("◈ PARALLAX").classes("text-xl font-bold")
        ui.label("Municipal Plan ↔ EU Grant Matching").classes("text-sm opacity-70")


def create_main_page():
    create_header()

    # plan selector
    plan_options = {p.id: f"{p.municipality} ({p.region}, {p.population:,})" for p in plans}
    selected_plan = ui.select(plan_options, value=plans[0].id, label="Select municipal plan").classes("w-96")

    results_container = ui.column().classes("w-full mt-4")

    def run_matching():
        results_container.clear()
        plan_id = selected_plan.value
        plan = get_plan_by_id(plan_id)
        results = matcher.match_plan(plan_id, top_k=8)
        confidence = analyze_score_distribution(results)
        gaps = analyze_gaps(plan_id)
        coverage = get_plan_coverage(plan_id)

        with results_container:

            # confidence summary
            color = {"strong": "green", "moderate": "yellow", "ambiguous": "orange", "weak": "red"}.get(confidence["quality"], "gray")
            with ui.card().classes("w-full mb-4"):
                ui.label("Result confidence").classes("text-lg font-bold")
                ui.badge(confidence["quality"].upper(), color=color)
                ui.label(confidence["summary"]).classes("text-sm mt-1")

            # coverage bar
            with ui.card().classes("w-full mb-4"):
                ui.label("Domain coverage").classes("text-lg font-bold")
                ratio = coverage["coverage_ratio"]
                ui.linear_progress(value=ratio).classes("mt-2")
                ui.label(f"{len(coverage['covered'])}/{len(coverage['covered']) + len(coverage['uncovered'])} domains covered ({ratio:.0%})").classes("text-sm")

            # matches
            ui.label("Top matches").classes("text-xl font-bold mt-4")

            for r in results:
                grant = get_grant_by_id(r.grant_id, grants)
                if not grant:
                    continue

                conf_color = {"high": "green", "medium": "yellow", "low": "red"}[r.confidence]
                budget = f"€{grant.budget_eur:,}" if grant.budget_eur else "N/A"

                with ui.card().classes("w-full mb-2"):
                    with ui.row().classes("items-center justify-between w-full"):
                        with ui.row().classes("items-center gap-2"):
                            ui.badge(f"#{r.rank}", color="primary")
                            ui.label(grant.name).classes("font-bold")
                        with ui.row().classes("items-center gap-2"):
                            ui.badge(f"{r.similarity_score:.3f}", color=conf_color)
                            ui.badge(r.confidence, color=conf_color, outline=True)

                    ui.label(f"{grant.programme} · {budget} · deadline {grant.deadline}").classes("text-sm opacity-70")

                    if r.shared_themes:
                        with ui.row().classes("gap-1 mt-1"):
                            for theme in r.shared_themes:
                                ui.badge(theme.replace("_", " "), color="teal", outline=True).classes("text-xs")

                    # expandable explanations
                    with ui.expansion("Explanations").classes("w-full mt-2"):
                        with ui.tabs().classes("w-full") as tabs:
                            citizen_tab = ui.tab("Citizen")
                            analyst_tab = ui.tab("Analyst")
                            auditor_tab = ui.tab("Auditor")

                        with ui.tab_panels(tabs, value=citizen_tab).classes("w-full"):
                            with ui.tab_panel(citizen_tab):
                                ui.markdown(rag.explain(r, level="citizen"))
                            with ui.tab_panel(analyst_tab):
                                ui.markdown(rag.explain(r, level="analyst").replace("\n", "\n\n"))
                            with ui.tab_panel(auditor_tab):
                                ui.code(rag.explain(r, level="auditor"))

            # gap analysis
            if gaps:
                ui.label("Funding gaps").classes("text-xl font-bold mt-6")
                ui.label("Domains not in your plan that could unlock additional funding:").classes("text-sm opacity-70 mb-2")

                for gap in gaps[:6]:
                    label = gap.domain.replace("_", " ").title()
                    budget = f"€{gap.estimated_budget:,}"
                    with ui.card().classes("w-full mb-2"):
                        with ui.row().classes("items-center justify-between w-full"):
                            ui.label(label).classes("font-bold")
                            ui.badge(f"{len(gap.potential_grants)} grants · {budget}", color="orange")
                        ui.label(gap.recommendation).classes("text-sm mt-1")

            # what-if
            ui.label("What-If explorer").classes("text-xl font-bold mt-6")
            custom_text = ui.textarea(
                label="Modify plan text and see how matches change",
                value=plan.full_text[:500],
            ).classes("w-full")

            whatif_results = ui.column().classes("w-full mt-2")

            def run_whatif():
                whatif_results.clear()
                text = custom_text.value
                if not text.strip():
                    return
                new_results = matcher.match_text(text, top_k=5)
                with whatif_results:
                    for r in new_results:
                        g = get_grant_by_id(r.grant_id, grants)
                        if not g:
                            continue
                        conf_color = {"high": "green", "medium": "yellow", "low": "red"}[r.confidence]
                        with ui.row().classes("items-center gap-2 mb-1"):
                            ui.badge(f"#{r.rank}", color="primary")
                            ui.label(g.name).classes("text-sm")
                            ui.badge(f"{r.similarity_score:.3f}", color=conf_color)

            ui.button("Re-match", on_click=run_whatif).classes("mt-2")

    selected_plan.on("update:model-value", lambda: run_matching())
    run_matching()  # initial load


# launching launching
ui.dark_mode(True)
create_main_page()
ui.run(title="Parallax", port=8080)
