"""Generate offline fixture pages (fictional companies, fictional URLs) + search index + eval labels.

Run: python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://fixtures.mlia.local"

PAGES = [
    dict(slug="news/horizon-launches-palm-crest-phase-3", signal_type="new_launch", tags=["launch", "new cairo", "2026"],
         title="Horizon Developments launches Palm Crest Phase 3 in New Cairo",
         body=[
             "Horizon Developments launched Phase 3 of its Palm Crest compound in New Cairo on Sunday, adding 420 apartments and townhouses.",
             "Prices start from EGP 7.8 million for two-bedroom apartments, with a payment plan of 10% down over 8 years.",
             "The developer said delivery of the new phase is scheduled for 2029.",
             "The company's chairman told reporters that demand in the Fifth Settlement remains strong among end users.",
         ],
         label=dict(signal_type="new_launch", entity="Horizon Developments", area="New Cairo", payment_plan=True,
                    budget_band="6-10M EGP")),
    dict(slug="markets/palm-crest-phase-3-sales", signal_type="new_launch", tags=["launch", "new cairo"],
         title="Palm Crest Phase 3 opens for sales",
         body=[
             "Horizon Development launched the third phase of Palm Crest in New Cairo this week.",
             "Brokers report strong early interest in the townhouse units.",
         ],
         label=dict(signal_type="new_launch", entity="Horizon Developments", area="New Cairo")),
    dict(slug="news/nile-crest-raises-prices-zayed", signal_type="price_change",
         tags=["price increase", "sheikh zayed", "2026", "payment plan"],
         title="Nile Crest Group raises prices 12% at Zayed Dunes",
         body=[
             "Nile Crest Group raised prices at its Zayed Dunes project in Sheikh Zayed by 12% effective December 1, 2026.",
             "The new payment plan requires 15% down with installments over 7 years, compared with 10 years previously.",
             "Price per square meter now averages EGP 82,000.",
         ],
         label=dict(signal_type="price_change", entity="Nile Crest Group", area="Sheikh Zayed", payment_plan=True)),
    dict(slug="business/sahara-urban-new-capital-plan", signal_type="price_change",
         tags=["payment plan", "new capital", "installments", "years", "down payment"],
         title="Sahara Urban extends installments to 12 years in the New Capital",
         body=[
             "Sahara Urban introduced a new payment plan at Capital Gardens in the New Administrative Capital: 5% down payment with installments over 12 years.",
             "The offer runs until the end of the year for apartments priced from EGP 4.2 million.",
         ],
         label=dict(signal_type="price_change", entity="Sahara Urban", area="New Capital", payment_plan=True,
                    budget_band="3-6M EGP")),
    dict(slug="tech/finovate-opens-cairo-hub", signal_type="office_opening",
         tags=["company", "opens", "office", "new cairo", "egypt", "2026", "headquarters", "hiring"],
         title="Finovate Technologies opens regional hub in New Cairo, plans 300 hires",
         body=[
             "Finovate Technologies opens its new regional headquarters in New Cairo in January 2026, the fintech firm announced.",
             "The company plans to hire 300 engineers over two years and relocate 40 managers from Dubai and London.",
             "A spokesperson said the company will provide housing support for relocating employees.",
         ],
         label=dict(signal_type="office_opening", entity="Finovate Technologies", area="New Cairo")),
    dict(slug="business/delta-pharma-zayed-office", signal_type="office_opening",
         tags=["company", "office", "sheikh zayed", "expands", "egypt", "2026"],
         title="Delta Pharma expands its presence with new Sheikh Zayed offices",
         body=[
             "Delta Pharma inaugurated new administrative offices in Sheikh Zayed, expanding its presence in West Cairo.",
             "The site will host 150 employees, including regional staff transferring from Alexandria.",
         ],
         label=dict(signal_type="office_opening", entity="Delta Pharma", area="Sheikh Zayed")),
    dict(slug="hr/globalmove-egypt-relocation", signal_type="relocation_firm",
         tags=["relocation", "employee", "egypt", "cairo", "2026", "company"],
         title="GlobalMove Relocations sees surge in assignees moving to Cairo",
         body=[
             "GlobalMove Relocations said it is handling a record number of employees relocating to Cairo in 2026, mostly to New Cairo and Sheikh Zayed.",
             "The mobility firm is seeking local housing partners for furnished apartments.",
         ],
         label=dict(signal_type="relocation_firm", entity="GlobalMove Relocations", area="New Cairo")),
    dict(slug="portal/listingsplus-partnership", signal_type="competitor_move",
         tags=["egypt", "real estate", "news", "aqarmap", "property finder", "bayut"],
         title="ListingsPlus Egypt partners with banks on mortgage pre-approval",
         body=[
             "ListingsPlus Egypt rolls out a mortgage pre-approval feature in partnership with two local banks, the property portal said.",
             "The feature lets buyers check eligibility before contacting a broker.",
         ],
         label=dict(signal_type="competitor_move", entity="ListingsPlus Egypt", area=None)),
    dict(slug="society/private-gossip", signal_type="office_opening",
         tags=["company", "office", "new cairo", "2026"],
         title="Executive moves to New Cairo",
         body=[
             "The executive's wife and children moved into a villa in New Cairo last month, according to neighbours.",
             "He earns a salary of EGP 2 million per month, the report claimed.",
         ],
         label=dict(signal_type=None)),
    dict(slug="lifestyle/ramadan-recipes", signal_type="new_launch", tags=["new cairo", "2026"],
         title="Ten Ramadan recipes to try this year",
         body=["Our favourite recipes for the holy month include stuffed vine leaves and kunafa."],
         label=dict(signal_type=None)),
]

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body><header><nav>Home | Business | Real Estate</nav></header>
<article><h1>{title}</h1><p class="date">2026-10-28</p>
{paras}
</article><footer>Fixture page for offline tests. Fictional companies.</footer></body></html>
"""


def main() -> None:
    pages_dir = ROOT / "fixtures" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    index, labels = [], []
    for p in PAGES:
        url = f"{HOST}/{p['slug']}"
        (pages_dir / (p["slug"].replace("/", "_") + ".html")).write_text(
            TEMPLATE.format(title=p["title"], paras="\n".join(f"<p>{b}</p>" for b in p["body"])), encoding="utf-8")
        index.append({"url": url, "title": p["title"], "snippet": p["body"][0], "signal_type": p["signal_type"],
                      "tags": p["tags"]})
        labels.append({"url": url, **p["label"]})
    (ROOT / "fixtures" / "search_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "labeled_pages.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(f"wrote {len(PAGES)} fixture pages")


if __name__ == "__main__":
    main()
