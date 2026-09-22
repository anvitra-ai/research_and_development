"""21 template classes from Formica et al. (2023) for KB question answering."""

from __future__ import annotations

# Table 1 — logical, comparative, quantitative, and simple question classes.
FORMICA_TEMPLATE_CLASSES: list[str] = [
    "F_LogUnion",
    "F_LogIntersection",
    "F_LogDifference",
    "F_CompCountMore",
    "F_CompCountLess",
    "F_CompCountApprox",
    "F_CompMore",
    "F_CompLess",
    "F_CompApprox",
    "F_QuantCount",
    "F_QuantCountAtmost",
    "F_QuantCountAtleast",
    "F_QuantCountApprox",
    "F_QuantCountEqual",
    "F_QuantAtmost",
    "F_QuantAtleast",
    "F_QuantApprox",
    "F_QuantEqual",
    "F_QuantMax",
    "F_QuantMin",
    "F_Simple",
]

FORMICA_CLASS_LABELS: dict[str, str] = {
    "F_LogUnion": "Logical Union",
    "F_LogIntersection": "Logical Intersection",
    "F_LogDifference": "Logical Difference",
    "F_CompCountMore": "Comparative Count over More",
    "F_CompCountLess": "Comparative Count over Less",
    "F_CompCountApprox": "Comparative Count over Approx",
    "F_CompMore": "Comparative More",
    "F_CompLess": "Comparative Less",
    "F_CompApprox": "Comparative Approx",
    "F_QuantCount": "Quantitative Count",
    "F_QuantCountAtmost": "Quantitative Count over Atmost",
    "F_QuantCountAtleast": "Quantitative Count over Atleast",
    "F_QuantCountApprox": "Quantitative Count over Approx",
    "F_QuantCountEqual": "Quantitative Count over Equal",
    "F_QuantAtmost": "Quantitative Atmost",
    "F_QuantAtleast": "Quantitative Atleast",
    "F_QuantApprox": "Quantitative Approx",
    "F_QuantEqual": "Quantitative Equal",
    "F_QuantMax": "Quantitative Max",
    "F_QuantMin": "Quantitative Min",
    "F_Simple": "Simple Question",
}

# Discriminative keywords used in the paper's semantic feature extraction.
DISCRIMINATIVE_KEYWORDS: dict[str, list[str]] = {
    "F_LogUnion": ["or", "either", "union"],
    "F_LogIntersection": ["and", "both", "intersection", "also"],
    "F_LogDifference": ["but not", "without", "except", "excluding", "difference"],
    "F_CompCountMore": ["how many", "more than", "greater than", "higher than"],
    "F_CompCountLess": ["how many", "less than", "fewer than", "lower than"],
    "F_CompCountApprox": ["how many", "approximately", "around", "about"],
    "F_CompMore": ["more than", "greater", "higher", "larger", "bigger"],
    "F_CompLess": ["less than", "lower", "smaller", "fewer"],
    "F_CompApprox": ["approximately", "around", "similar", "about the same"],
    "F_QuantCount": ["how many", "how much", "count", "number of"],
    "F_QuantCountAtmost": ["at most", "atmost", "no more than"],
    "F_QuantCountAtleast": ["at least", "atleast", "minimum of"],
    "F_QuantCountApprox": ["how many", "approximately", "around"],
    "F_QuantCountEqual": ["exactly", "equal to", "precisely"],
    "F_QuantAtmost": ["at most", "atmost", "maximum", "up to"],
    "F_QuantAtleast": ["at least", "atleast", "minimum"],
    "F_QuantApprox": ["approximately", "around", "roughly"],
    "F_QuantEqual": ["exactly", "equal", "precisely"],
    # "most"/"least" deliberately excluded: "most recently reported quarter" is a
    # standard qualifier phrase in this dataset, not a superlative/max request, and
    # a bare substring match can't tell the two apart.
    "F_QuantMax": ["max", "maximum", "highest", "largest"],
    "F_QuantMin": ["min", "minimum", "lowest", "smallest"],
    "F_Simple": ["what", "who", "which", "how does", "does", "list", "name", "trace", "explain"],
}

# Domain relationship hints for triplet slot prop1.
# Original entries below are for kg/india_theme_kg.cypher's thematic (Hormuz/oil)
# graph and never match anything in the Graphiti-ingested banking KG -- kept as-is
# so a run against that graph still works. Banking-domain entries use Graphiti's
# actual edge_types keys (see graphiti_neo4j.ipynb) and are ranked ahead of any
# overlapping legacy keyword since infer_prop1() prefers the longest keyword match.
RELATION_KEYWORDS: dict[str, list[str]] = {
    "TRANSITS": ["transit", "passes through", "transits through"],
    "HURT_BY": ["hurt", "downside", "impact", "affect", "exposed"],
    "HEDGES": ["hedge", "hedging", "protect", "hedging strategy"],
    "BENEFITS_FROM": ["beneficiar", "benefit", "winner", "upside", "gain from"],
    "OWNS": ["own", "ownership", "stake", "subsidiary", "hold"],
    "INVESTS_IN": ["invest", "fund", "back", "funds"],
    "REGULATED_BY": ["regulat", "policy", "government", "intervention"],
    "CONSTRAINED_BY": ["constrain", "bottleneck", "capacity", "limit"],
    "CAUSES": ["cause", "lead to", "propagate", "flow into"],
    "PRICE_LINKED_TO": ["price", "cost base", "linked to", "track"],
    "CONTAINS": ["sector", "constituent", "in sector"],
    # --- Banking ontology (Graphiti edge_types) ---
    "GovernanceRole": ["ceo", "md & ceo", "managing director", "chairman", "chief executive", "cfo", "board member"],
    "OwnershipStake": ["promoter", "shareholder", "shareholding", "% held", "percent held", "held by", "stake in"],
    "RegulatoryRelation": ["regulated by", "regulator", "license", "licence"],
    "HAS_RATING": ["credit rating", "rated", "rating agency", "credit rated"],
    "RATED_BY": ["rating agency", "rated by"],
    "BELONGS_TO_SEGMENT": ["banking segment"],  # keep narrow: "segment does"/"which segment" also match unrelated "customer segment" questions
    "OFFERS_PRODUCT": ["product", "offers", "home loan", "deposit product"],
    "PARTNERS_WITH": ["partnership", "partners with", "partnered with"],
    "COMPETES_WITH": ["competitor", "competes with", "competition"],
    "CorporateStructureRelation": ["subsidiary of", "parent company", "corporate group"],
}

# Preferred entity types for triplet slot assignment (first match wins).
SLOT_SUBJECT_PRIORITY: dict[str, list[str]] = {
    "F_Simple": [
        "COMPANY", "SECTOR", "RAW_MATERIAL", "PRODUCT", "EVENT", "MACRO_VAR",
        "GEOGRAPHY", "INVESTOR",
    ],
    "F_LogUnion": ["COMPANY", "SECTOR", "EVENT", "MACRO_VAR", "GEOGRAPHY"],
    "F_LogIntersection": ["COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"],
    "F_LogDifference": ["COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"],
    "F_CompMore": ["COMPANY", "SECTOR"],
    "F_CompLess": ["COMPANY", "SECTOR"],
    "F_CompApprox": ["COMPANY", "SECTOR"],
    "F_CompCountMore": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_CompCountLess": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_CompCountApprox": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_QuantCount": ["RAW_MATERIAL", "PRODUCT", "COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_QuantCountAtmost": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_QuantCountAtleast": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_QuantCountApprox": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_QuantCountEqual": ["COMPANY", "SECTOR", "GEOGRAPHY"],
    "F_QuantAtmost": ["COMPANY", "SECTOR", "MACRO_VAR", "GEOGRAPHY"],
    "F_QuantAtleast": ["COMPANY", "SECTOR", "MACRO_VAR", "GEOGRAPHY"],
    "F_QuantApprox": ["COMPANY", "SECTOR", "MACRO_VAR", "GEOGRAPHY"],
    "F_QuantEqual": ["COMPANY", "SECTOR", "MACRO_VAR", "GEOGRAPHY"],
    "F_QuantMax": ["COMPANY", "SECTOR", "MACRO_VAR", "GEOGRAPHY"],
    "F_QuantMin": ["COMPANY", "SECTOR", "MACRO_VAR", "GEOGRAPHY"],
}

SLOT_SHOCK_PRIORITY: list[str] = ["EVENT", "MACRO_VAR", "GEOGRAPHY"]

FORMICA_NEEDED_TYPES: dict[str, set[str]] = {
    "F_LogUnion": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"},
    "F_LogIntersection": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"},
    "F_LogDifference": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"},
    "F_CompCountMore": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"},
    "F_CompCountLess": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"},
    "F_CompCountApprox": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR"},
    "F_CompMore": {"COMPANY", "SECTOR", "MACRO_VAR", "EVENT"},
    "F_CompLess": {"COMPANY", "SECTOR", "MACRO_VAR", "EVENT"},
    "F_CompApprox": {"COMPANY", "SECTOR", "MACRO_VAR", "EVENT"},
    "F_QuantCount": {"COMPANY", "SECTOR", "GEOGRAPHY", "RAW_MATERIAL", "PRODUCT"},
    "F_QuantCountAtmost": {"COMPANY", "SECTOR", "GEOGRAPHY"},
    "F_QuantCountAtleast": {"COMPANY", "SECTOR", "GEOGRAPHY"},
    "F_QuantCountApprox": {"COMPANY", "SECTOR", "GEOGRAPHY"},
    "F_QuantCountEqual": {"COMPANY", "SECTOR", "GEOGRAPHY"},
    "F_QuantAtmost": {"COMPANY", "SECTOR", "MACRO_VAR"},
    "F_QuantAtleast": {"COMPANY", "SECTOR", "MACRO_VAR"},
    "F_QuantApprox": {"COMPANY", "SECTOR", "MACRO_VAR"},
    "F_QuantEqual": {"COMPANY", "SECTOR", "MACRO_VAR"},
    "F_QuantMax": {"COMPANY", "SECTOR", "MACRO_VAR", "EVENT"},
    "F_QuantMin": {"COMPANY", "SECTOR", "MACRO_VAR", "EVENT"},
    "F_Simple": {"COMPANY", "SECTOR", "GEOGRAPHY", "EVENT", "MACRO_VAR", "INVESTOR", "RAW_MATERIAL", "PRODUCT"},
}
