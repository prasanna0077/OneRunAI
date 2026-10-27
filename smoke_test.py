"""Headless self-test (no Streamlit needed): data -> MVP 1 -> MVP 2 -> MVP 3 -> Deal context + narrative.

PyCharm: right-click -> Run 'smoke_test'.  Prints a short health report.
"""
import time

from onerun import services
from onerun.ai.narrative import get_engine
from onerun.data_generation.generator import data_ready, generate_all


def main() -> None:
    t0 = time.time()
    if not data_ready():
        generate_all(progress=print)
    suite, scored = services.acquisition_results()
    print("\n[MVP 1] Acquisition models (hold-out):")
    print(suite.metrics.round(3).to_string(index=False))
    print("\nTier mix:", scored.tier.value_counts().to_dict())
    print(scored[["rank", "bank_name", "bank_type", "acquisition_score", "tier"]].head(5).to_string(index=False))

    bank = services.top_target_id()
    integ = services.integration_results(bank)
    print(f"\n[MVP 2] Integration for {bank}: complexity {integ.complexity['score']:.0f}/100 ({integ.complexity['band']})")
    print(integ.segments[["segment", "customers", "wave"]].to_string(index=False))
    print("Branch actions:", integ.branches.recommended_action.value_counts().to_dict())
    print("Product actions:", integ.products.recommended_action.value_counts().to_dict())

    rs = services.retention_suite()
    print("\n[MVP 3] Churn models (hold-out):")
    print(rs.metrics.round(3).to_string(index=False))
    ret = services.retention_scored(bank)
    print("Risk bands:", ret.risk_band.value_counts().to_dict(), "| avg churn prob:", round(ret.churn_probability.mean(), 3))

    ctx = services.deal_context(bank)
    print("\n[Deal context KPIs]", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in ctx["kpis"].items()})
    print("\n[Executive narrative]\n" + get_engine().generate(ctx))
    print(f"\nAll good - {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
