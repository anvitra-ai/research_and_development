"""Generate a synthetic dataset for DeBERTa query-template classification (stock impact analysis).

LEGACY: predates the Formica template classifier (scripts/train_template_classifier.py
is its replacement). Not imported by the current batch pipeline or API; kept for
reference only. Its `from paths import ...` import is stale and won't resolve as-is.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from paths import DATA_DIR

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

COMPANIES = [
    "ICICI Lombard", "Reliance Industries", "Indian Oil", "ONGC", "Bharat Petroleum",
    "Tata Motors", "Infosys", "State Bank of India", "Adani Ports", "NTPC",
    "Hindustan Unilever", "Larsen & Toubro", "Petronet LNG", "GAIL", "Muthoot Finance",
    "Asian Paints", "Bajaj Finance", "Cholamandalam Investment", "Great Eastern Shipping",
    "Cochin Shipyard", "Oil India", "Hindustan Petroleum", "Mangalore Refinery",
    "InterGlobe Aviation", "SpiceJet", "Shriram Finance", "Federal Bank", "CSB Bank",
    "New India Assurance", "GIC Re", "Coromandel International", "Chambal Fertilisers",
    "Tata Power", "Adani Green Energy", "Coal India", "Power Grid Corporation",
    "Bharat Electronics", "Solar Industries", "Netweb Technologies", "Voltamp Transformers",
]

GEOGRAPHIES = [
    "Strait of Hormuz", "Gulf producers", "Qatar", "India", "Red Sea",
    "Cape of Good Hope", "Oman", "Saudi Arabia", "UAE", "Russia",
]

EVENTS = [
    "Hormuz disruption", "oil supply shock", "LNG curtailment", "marine war-risk spike",
    "crude price rally", "geopolitical escalation", "Strait of Hormuz closure",
    "tanker freight surge", "LPG import disruption", "refinery outage",
]

MACRO_VARS = [
    "Brent crude price", "USD/INR", "RBI policy rate", "India CPI",
    "tanker freight rates", "marine war-risk premium", "gross refining margin",
    "spot LNG price", "current account deficit", "fertiliser subsidy bill",
    "Singapore GRM", "bond yields", "INR depreciation",
]

SECTORS = [
    "Oil & Gas", "Banks & NBFCs", "Aviation", "Fertilisers", "Shipping & Logistics",
    "Power & Utilities", "Defence", "Insurance", "Refining", "Metals & Mining",
    "IT Services", "Capital Goods", "Paints & Adhesives", "Telecom & Connectivity",
]

RAW_MATERIALS = [
    "crude oil", "LPG", "LNG", "naphtha", "urea", "ammonia", "sulphur",
    "base oil", "PVC resin", "carbon black", "aviation turbine fuel",
]

INVESTORS = [
    "Blackstone", "ADIA", "GIC", "Temasek", "SoftBank", "KKR", "Warburg Pincin",
    "CPP Investments", "Mubadala", "QIA",
]

TEMPLATE_EXAMPLES = {
    "T_StockImpact": [
        "Why are {company} stocks impacted by {event}?",
        "How does {event} affect {company} share price?",
        "What is the stock impact of {event} on {company}?",
        "Explain how {geo} disruption hits {company} equity.",
        "Is {company} vulnerable to {event}?",
        "Why is {company} moving on {macro} news?",
        "How exposed is {company} to {event}?",
        "Does {event} hurt {company} investors?",
        "What drives {company} selloff after {geo} tension?",
        "Why are {company} shares under pressure from {macro}?",
        "How does {geo} risk transmit to {company} stock?",
        "What is {company}'s equity sensitivity to {event}?",
        "Should investors worry about {company} after {event}?",
        "Why did {company} fall after {geo} headlines?",
        "What equity risk does {event} create for {company}?",
        "How material is {event} for {company} shareholders?",
        "Is {company} a casualty of {macro} volatility?",
        "What is the downside for {company} from {event}?",
    ],
    "T_CausalChain": [
        "How does {geo} disruption cause changes in {macro}?",
        "What is the causal chain from {event} to {macro}?",
        "Trace the path from {geo} to {company} via {macro}.",
        "How does {event} lead to higher {macro}?",
        "What links {geo} closure to {sector} stocks?",
        "Show transmission from {raw} shock to {company}.",
        "How does {macro} feed into {sector} earnings?",
        "What is the mechanism from {event} to bank stocks?",
        "How does {geo} affect {raw} and then {company}?",
        "Map causality from {event} to insurance stocks.",
        "Explain the knock-on effect from {geo} to {macro}.",
        "What is the transmission channel from {event} to {sector}?",
        "How does a shock at {geo} propagate to {company}?",
        "Walk through the causal links from {raw} to {macro}.",
        "What intermediate nodes connect {event} and {company}?",
        "How does disruption at {geo} cascade into {sector}?",
    ],
    "T_Beneficiary": [
        "Which companies benefit from {event}?",
        "Who gains when {macro} rises?",
        "Which {sector} names are beneficiaries of {macro}?",
        "Who benefits from higher {macro}?",
        "List stocks that gain from {geo} rerouting.",
        "Which insurers benefit from war-risk premium spike?",
        "Who wins from {event} in Indian markets?",
        "Which refiners benefit from {macro} widening?",
        "Name beneficiaries in {sector} from {event}.",
        "Which companies have positive exposure to {macro}?",
        "Who are the upside names if {event} persists?",
        "Which stocks rally when {macro} increases?",
        "Who profits from higher {macro} in {sector}?",
        "List Indian winners from {geo} disruption.",
        "Which names are net beneficiaries of {event}?",
    ],
    "T_SectorExposure": [
        "Which companies are in {sector}?",
        "List {sector} constituents exposed to {event}.",
        "Who in {sector} is most exposed to {geo}?",
        "Map {sector} exposure to {macro}.",
        "Which {sector} stocks are in the Hormuz theme?",
        "Show sector basket for {event} impact.",
        "What {sector} names are linked to {raw}?",
        "Break down {sector} sensitivity to {macro}.",
        "Which NBFCs sit in the impact chain for {company}?",
        "Give me {sector} players affected by oil shock.",
        "Enumerate {sector} companies tied to {event}.",
        "What is the {sector} peer set for {company}?",
        "Which {sector} stocks should I screen for {geo} risk?",
        "Show all {sector} names with {macro} linkage.",
    ],
    "T_Hedging": [
        "How does {company} hedge against {macro}?",
        "What natural hedge does {company} have for {event}?",
        "Does {company} offset {macro} risk via FX hedges?",
        "How is {company} protected from {geo} disruption?",
        "Which hedges reduce {company} oil exposure?",
        "Does {company} hedge {raw} price volatility?",
        "What risk mitigants does {company} use for {event}?",
        "How does {company} manage {macro} sensitivity?",
        "Is {company} hedged for marine war-risk?",
        "Explain hedging strategy of {company} for {macro}.",
        "What derivative hedges does {company} run on {macro}?",
        "Does {company} have a natural offset to {event}?",
        "How much of {company}'s {macro} risk is hedged?",
        "What protects {company} margins from {raw} spikes?",
    ],
    "T_PriceLinkage": [
        "How is {company} linked to {macro}?",
        "What is the price linkage between {raw} and {company}?",
        "Does {company} track {macro}?",
        "How correlated is {company} with {macro}?",
        "Which input prices move with {macro} for {company}?",
        "Is {company} realization tied to Brent?",
        "How does {raw} price affect {company} margins?",
        "What commodities drive {company} cost base?",
        "Link {company} earnings to {macro}.",
        "How does naphtha price impact {company}?",
        "Is {company} pricing indexed to {macro}?",
        "What is the beta of {company} to {macro}?",
        "How closely does {company} follow {raw} prices?",
        "Does {company} pass through {macro} changes?",
    ],
    "T_TransitRisk": [
        "How much Indian {raw} transits through {geo}?",
        "What share of imports passes {geo}?",
        "How dependent is India on {geo} for {raw}?",
        "Which routes are affected if {geo} closes?",
        "What is transit exposure of {raw} via {geo}?",
        "How does {geo} chokepoint impact Indian energy imports?",
        "What percentage of LNG flows through {geo}?",
        "Is {company} exposed to {geo} shipping routes?",
        "How does rerouting via Cape affect {company}?",
        "What is Hormuz transit risk for Indian {raw}?",
        "How much of India's {raw} import basket uses {geo}?",
        "What import share transits {geo} for {raw}?",
        "If {geo} shuts, how much {raw} supply is at risk?",
        "Quantify India's {raw} dependence on {geo}.",
    ],
    "T_SupplyDisruption": [
        "How does {event} disrupt {raw} supply?",
        "What supply chain risk does {geo} pose for {company}?",
        "Which inputs become scarce after {event}?",
        "How does {company} source {raw} during disruption?",
        "What supply bottlenecks hit {sector} from {event}?",
        "Does {company} face feedstock shortage from {geo}?",
        "How does LNG curtailment affect {company}?",
        "What procurement risk does {event} create for {company}?",
        "Which suppliers are disrupted by {geo} tension?",
        "How does {raw} shortage propagate to {sector}?",
        "Can {company} secure {raw} if {geo} closes?",
        "What supply interruptions follow {event}?",
        "How vulnerable is {company}'s {raw} sourcing to {geo}?",
        "Which feedstocks are at risk from {event}?",
    ],
    "T_MacroTransmission": [
        "How does {macro} affect {company}?",
        "What happens to {company} when {macro} rises?",
        "How does {macro} flow into {sector} valuations?",
        "Does {macro} weaken {company} fundamentals?",
        "How does INR depreciation impact {company}?",
        "What is RBI rate impact on {company}?",
        "How does CPI shock affect {sector} demand?",
        "Does CAD widening hurt {company}?",
        "How do freight rates transmit to {company} costs?",
        "What macro channel links {event} to {company}?",
        "How do {macro} moves show up in {company} P&L?",
        "What earnings line moves first when {macro} shifts?",
        "How sensitive is {company} to a {macro} spike?",
        "Does a weaker rupee help or hurt {company}?",
    ],
    "T_RegulatoryImpact": [
        "How is {company} affected by fuel subsidy policy?",
        "What regulatory risk does {company} face in {sector}?",
        "How do LPG subsidy rules impact {company}?",
        "Is {company} regulated on {raw} pricing?",
        "What policy response affects {company} after {event}?",
        "How does excise duty change hit {company}?",
        "Which regulations constrain {company} margins?",
        "How does government intervention affect {sector}?",
        "What policy tailwinds support {company}?",
        "How is fertiliser subsidy linked to {company}?",
        "What government action caps {company} pricing?",
        "How do subsidy revisions change {company} outlook?",
        "Which policy levers matter for {company} in {sector}?",
        "What regulatory overhang exists for {company}?",
    ],
    "T_InvestmentFlow": [
        "Who invests in {company}?",
        "Which investors are backing AI infra plays?",
        "What capital flows into {sector} from {investor}?",
        "Does {investor} have exposure to Indian {sector}?",
        "Which PE firms invest in data centre ecosystem?",
        "Track investment from {investor} into Indian tech infra.",
        "Who funds {company} expansion?",
        "What FII flows affect {company}?",
        "Which sovereign funds invest in Indian {sector}?",
        "Map {investor} investments in Indian markets.",
        "Who are the major backers of {company}?",
        "What PE capital is entering Indian {sector}?",
        "Which global funds are buying into {company}?",
        "Where is {investor} deploying capital in India?",
    ],
    "T_OwnershipStructure": [
        "Who owns {company}?",
        "What subsidiaries does {company} have?",
        "Map corporate structure of {company}.",
        "Which group company controls {company}?",
        "What JVs does {company} operate?",
        "Who is the parent of {company}?",
        "List subsidiaries under {company}.",
        "How is {company} linked to ONGC group?",
        "What ownership ties connect refiners and marketing companies?",
        "Does {company} hold stakes in other listed entities?",
        "What is the promoter shareholding in {company}?",
        "Which group entities sit under {company}?",
        "Show ownership map for {company}.",
        "Who are the major shareholders of {company}?",
    ],
    "T_ConstraintRisk": [
        "What constrains {company} growth in AI infra?",
        "Which bottlenecks limit {sector} expansion?",
        "How does GPU supply constrain data centre buildout?",
        "What power availability risk hits {company}?",
        "Which constraints gate {sector} capex?",
        "How does transmission capacity limit {company}?",
        "What land or water constraints affect {company}?",
        "Which supply bottlenecks hurt {sector}?",
        "How does CRGO steel shortage impact transformers?",
        "What infrastructure constraints affect {company}?",
        "What capacity bottlenecks slow {company} rollout?",
        "Which physical constraints bind {sector} growth?",
        "How does grid evacuation limit {company}?",
        "What resource shortages cap {sector} expansion?",
    ],
    "T_CompareExposure": [
        "Compare {company} and Indian Oil exposure to {event}.",
        "Which is more exposed: {company} or Reliance Industries?",
        "Rank {sector} stocks by {macro} sensitivity.",
        "Compare insurers vs banks on Hormuz risk.",
        "Who is more hurt by Brent rally: {company} or Tata Motors?",
        "Compare refining vs marketing exposure to {event}.",
        "Which {sector} name has highest {geo} risk?",
        "Benchmark {company} against peers on {macro} linkage.",
        "Compare ONGC vs Oil India on Brent exposure.",
        "Which NBFC is more sensitive to {macro}?",
        "Between {company} and ONGC, who faces more {event} risk?",
        "Which peer has greater {macro} exposure: {company} or GAIL?",
        "Rank {company} versus sector peers on {geo} risk.",
        "Who is the more defensive name: {company} or {company}?",
    ],
    "T_EventScenario": [
        "What if {geo} closes completely?",
        "Scenario analysis for {event} on Nifty energy.",
        "What happens if Brent hits $120 again?",
        "Stress test {company} for prolonged {event}.",
        "Model escalation scenario for {geo}.",
        "What if LNG imports from Qatar are cut?",
        "Downside scenario for {sector} from {event}.",
        "What if war-risk premium stays at 5%?",
        "Simulate de-escalation impact on {macro}.",
        "What if India reroutes all cargo via Cape?",
        "Run a bear case for {company} under {event}.",
        "What if {geo} disruption lasts six months?",
        "Model base and bull cases for {sector} after {event}.",
        "What happens to markets if {macro} doubles?",
    ],
}

SLOT_VALUES = {
    "company": COMPANIES,
    "geo": GEOGRAPHIES,
    "event": EVENTS,
    "macro": MACRO_VARS,
    "sector": SECTORS,
    "raw": RAW_MATERIALS,
    "investor": INVESTORS,
}


def fill_template(pattern: str) -> str:
    text = pattern
    for slot, values in SLOT_VALUES.items():
        placeholder = "{" + slot + "}"
        picks = []
        while placeholder in text:
            if slot == "company" and len(picks) >= 1:
                # avoid duplicate company names in compare-style templates
                pool = [v for v in values if v not in picks] or values
                pick = random.choice(pool)
            else:
                pick = random.choice(values)
            picks.append(pick)
            text = text.replace(placeholder, pick, 1)
    return text


def build_dataset(examples_per_template: int) -> pd.DataFrame:
    rows = []
    for label, patterns in TEMPLATE_EXAMPLES.items():
        generated = set()
        attempts = 0
        max_attempts = examples_per_template * 50
        while len(generated) < examples_per_template and attempts < max_attempts:
            pattern = random.choice(patterns)
            text = fill_template(pattern)
            if text not in generated:
                generated.add(text)
                rows.append({"text": text, "label": label})
            attempts += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows, test_rows = [], []
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label].sample(frac=1, random_state=RANDOM_SEED)
        split = max(1, int(len(subset) * (1 - test_ratio)))
        train_rows.append(subset.iloc[:split])
        test_rows.append(subset.iloc[split:])
    train_df = pd.concat(train_rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1, random_state=RANDOM_SEED + 1).reset_index(drop=True)
    return train_df, test_df


def main(examples_per_template: int = 100, test_ratio: float = 0.2) -> None:
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(examples_per_template=examples_per_template)
    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_df, test_df = stratified_split(df, test_ratio=test_ratio)

    train_df.to_csv(out_dir / "deberta_stock_impact_train.csv", index=False)
    test_df.to_csv(out_dir / "deberta_stock_impact_test.csv", index=False)

    meta = {
        "task": "stock_impact_query_template_classification",
        "num_labels": len(labels),
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "examples_per_template": examples_per_template,
        "test_ratio": test_ratio,
    }
    (out_dir / "deberta_stock_impact_labels.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(train_df)} train / {len(test_df)} test examples")
    print(f"Per-class train counts:\n{train_df['label'].value_counts().sort_index().to_string()}")
    print(f"\nPer-class test counts:\n{test_df['label'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-per-template", type=int, default=100)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(examples_per_template=args.examples_per_template, test_ratio=args.test_ratio)
