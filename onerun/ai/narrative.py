"""Narrative engine - the seam where GenAI plugs in (Phase 2 Executive Intelligence, Phase 3 Deal Assistant).

Today: a deterministic, template-based engine (no API key needed).
Later : implement LLMNarrativeEngine.generate() and return it from get_engine().
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from onerun.config import ASSUMPTIONS, THRESHOLDS


class NarrativeEngine(ABC):
    @abstractmethod
    def generate(self, ctx: dict) -> str:
        """ctx = services.deal_context(bank_id). Returns markdown."""


def verdict_rules(k: dict) -> list[tuple[str, bool]]:
    """The three tests behind a 'Proceed' verdict, each as (description, passed)."""
    return [
        (f"Acquisition tier is Prime Target (score ≥ {ASSUMPTIONS['tier_prime']}) - this deal: {k['tier']}", k["tier"] == "Prime Target"),
        (f"Integration complexity is not High (< {THRESHOLDS['complexity_high']}) - this deal: {k['complexity_score']:.0f} ({k['complexity_band']})",
         k["complexity_band"] != "High"),
        (f"Expected churn is below {THRESHOLDS['churn_limit_pct']:.0f}% - this deal: {k['expected_churn_pct']:.1f}%",
         k["expected_churn_pct"] < THRESHOLDS["churn_limit_pct"]),
    ]


def recommendation(k: dict) -> tuple[str, str]:
    if all(ok for _, ok in verdict_rules(k)):
        return "PROCEED", "Strong fundamentals, manageable integration and contained customer risk."
    if k["tier"] in ("Prime Target", "Strong Candidate"):
        return "PROCEED WITH CONDITIONS", "Attractive target, but integration complexity or churn risk needs a mitigation plan."
    if k["tier"] == "Watchlist":
        return "HOLD", "Fundamentals are mixed - run deeper due diligence before committing."
    return "DO NOT PROCEED", "The model sees a low probability of synergy realisation."


class TemplateNarrativeEngine(NarrativeEngine):
    @staticmethod
    def _combined(ctx: dict) -> str:
        res = ctx.get("combined")
        if not res:
            return ""
        from onerun.ml.proforma import summary_lines
        lines = summary_lines(res, ctx["acquirer"], ctx["target"].bank_name)
        return "**Combined bank (Day-1)** - " + lines[0].replace("Combined balance sheet: ", "") + " " + " ".join(lines[2:4]) + "\n\n"

    def generate(self, ctx: dict) -> str:
        t, k, integ, ret = ctx["target"], ctx["kpis"], ctx["integration"], ctx["retention"]
        verdict, why = recommendation(k)
        flags = "" if t.red_flags == "None" else f"\n- **Red flags:** {t.red_flags}"
        top_reason = ret.lead_reason.value_counts().idxmax() if ret.lead_reason.notna().any() else "n/a"
        wave3 = float((integ.customers.migration_wave >= 3).mean() * 100)
        return (
            f"### Recommendation: {verdict}\n{why}\n\n"
            f"**Strategic fit** - {t.bank_name} ({t.bank_type}, {t.region}) ranks **#{k['rank']}** of {len(ctx['scored'])} candidates with an "
            f"acquisition score of **{k['acquisition_score']:.0f}/100** ({k['tier']}). Risk-adjusted synergy is about "
            f"**₹{k['risk_adj_synergy_cr']:,.0f} Cr a year** against an indicative valuation of ₹{k['valuation_cr']:,.0f} Cr.{flags}\n\n"
            f"**Integration outlook** - complexity is **{k['complexity_band'].lower()} ({k['complexity_score']:.0f}/100)**. "
            f"{wave3:.0f}% of customers need relationship-led or assisted migration; branch consolidation could release "
            f"≈ ₹{k['branch_saving_cr']:,.0f} Cr a year.\n\n"
            f"**Customer risk** - expected 12-month churn is **{k['expected_churn_pct']:.1f}%**, with {k['high_risk_customers']} high-risk "
            f"customers in the scored sample and ≈ ₹{k['revenue_at_risk_cr']:,.0f} Cr of annual revenue at risk. "
            f"The most common driver is *{top_reason}*.\n\n"
            f"{self._combined(ctx)}"
            "**Next steps** - (1) confirm findings in due diligence, (2) lock the migration-wave plan, "
            "(3) fund the retention playbook before day-1."
        )


class LLMNarrativeEngine(NarrativeEngine):
    """Phase 3 placeholder. Build a prompt from `ctx`, call your LLM of choice, return markdown."""

    def generate(self, ctx: dict) -> str:
        raise NotImplementedError("LLM narrative engine is a Phase 3 feature - see README 'Roadmap hooks'.")


def get_engine() -> NarrativeEngine:
    return TemplateNarrativeEngine()
