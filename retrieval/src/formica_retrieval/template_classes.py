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
    # New: cross-company / group-aggregation family. None of the classes above
    # can answer a query that isn't about one company or a fixed pair -- "across
    # all 39 banks, which reports the highest X", "which segment has the highest
    # average X", "X's ratio vs. the average for its segment" all need to rank or
    # aggregate over the whole company universe (or a BankingSegment group of it).
    "F_GlobalRank",           # single company ranks highest/lowest on X across all companies (optionally segment-filtered)
    "F_GroupAggregate",       # AVG(X) per BankingSegment -- either "which segment ranks highest" or segment-vs-segment
    "F_CompToGroupAverage",   # one company's X vs. AVG(X) over its own segment's other members
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
    "F_GlobalRank": "Global Rank Across Companies",
    "F_GroupAggregate": "Group/Segment Aggregate",
    "F_CompToGroupAverage": "Compare to Group Average",
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
# Shared vocabulary for financial-metric queries -- reused below on several
# "narrative" relation names (HAS_WEAKNESS, HAS_STRENGTH, ...) that frequently
# carry a specific figure the query asks about, in addition to HAS_METRIC/
# MetricObservation. See the comment further down for why.
_METRIC_TERMS = [
    "ratio", "casa", "nim", "net interest margin", "roa", "roe",
    "return on asset", "return on equity", "npa", "non-performing",
    "capital adequacy", "crar", "cet1", "advances", "loan book",
    "deposit growth", "deposits", "profit", "income", "margin",
]

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
    "HOLDS_ROLE": ["ceo", "md & ceo", "managing director", "chairman", "chief executive", "cfo", "board member"],
    # Ownership is stored under three different relation-type names across ingestion
    # episodes (schema drift, not a single canonical name) -- all three must be in
    # prop1_list or a promoter/shareholder query silently misses whichever name the
    # fact for that particular company happens to use.
    "OwnershipStake": ["promoter", "shareholder", "shareholding", "% held", "percent held", "held by", "stake in"],
    "HELD_BY": ["promoter", "shareholder", "shareholding", "% held", "percent held", "held by", "stake in"],
    "OWNS_STAKE_IN": ["promoter", "shareholder", "shareholding", "% held", "percent held", "held by", "stake in"],
    "RegulatoryRelation": ["regulated by", "regulator", "license", "licence"],
    "REGULATORY_RELATION": ["regulated by", "regulator", "license", "licence"],
    # HAS_METRIC/MetricObservation carry virtually every numeric KPI a query asks
    # about (CASA, NIM, ROA/ROE, NPA ratios, CRAR/CET1, advances, deposits, ...),
    # split across two synonymous relation names depending on ingestion episode.
    # Neither was in this map at all -- every metric-lookup query had an empty
    # prop1_list and fell straight into the unfiltered relation dump.
    "HAS_METRIC": [
        "ratio", "casa", "nim", "net interest margin", "roa", "roe",
        "return on asset", "return on equity", "npa", "non-performing",
        "capital adequacy", "crar", "cet1", "advances", "loan book",
        "deposit growth", "deposits", "profit", "income", "margin",
    ],
    "MetricObservation": [
        "ratio", "casa", "nim", "net interest margin", "roa", "roe",
        "return on asset", "return on equity", "npa", "non-performing",
        "capital adequacy", "crar", "cet1", "advances", "loan book",
        "deposit growth", "deposits", "profit", "income", "margin",
    ],
    # "Advances (loan book)" and "deposit growth" are frequently NOT modeled as
    # HAS_METRIC/MetricObservation at all -- confirmed directly against the
    # graph: ExposureRelation carries 5/31 advances-related edges and FUNDED_BY
    # carries 12/64 deposit-related edges, both entirely missed by the metric
    # keywords above (every "total advances and deposit growth" query was
    # narrowing to the wrong Metric node's uuid via target_ids AND missing
    # these relation names in prop1_list at the same time -- fixing only one
    # of the two would still have left this query family broken). Scoped
    # narrowly to just advances/deposit phrasing, not the full metric keyword
    # list above, since ExposureRelation/FUNDED_BY are also used for other,
    # unrelated things (general risk exposure, other funding sources).
    # ExposureRelation is a wrapper class whose *stored* r.name is always the
    # literal string "ExposureRelation" -- the semantic sub-type it documents
    # (HAS_ASSET_EXPOSURE_TO / HAS_LOAN_EXPOSURE_TO / EXPOSED_TO) never actually
    # appears as an r.name value in the graph. The "EXPOSED_TO" keyword entry
    # below was therefore dead on arrival (confirmed: 0 matches for SBI's
    # credit/interest-rate/cyber/regulatory risk facts, which are all really
    # stored as ExposureRelation) -- merged its risk/weakness keywords in here
    # instead, alongside the advances/loan-book ones already scoped to this
    # same relation name for the unrelated "total advances" query family.
    # "product"/"service"/"loan" merged in here too: "what products and services
    # does X offer" queries were only matching OFFERS_PRODUCT/PROVIDES_SERVICE,
    # but the single biggest source of real product facts (31 edges across the
    # dataset -- every per-loan-type fact: "State Bank of India offers home
    # loans", auto/personal/education/agricultural/MSME loans, etc.) is filed
    # under the ExposureRelation wrapper name like everything else that class
    # touches, so those answers were being silently dropped despite being
    # exactly on-topic.
    "ExposureRelation": [
        "advances", "loan book", "advance growth",
        "weakness", "vulnerability", "vulnerable", "risk", "risks", "exposed to", "exposure to",
        "product", "service", "loan", "loans", "offer",
    ],
    "FUNDED_BY": ["deposit growth", "deposits", "deposit base"],
    # "rating agency" (singular) never substring-matches a query that asks "...
    # and from which rating AGENCIES?" (plural) -- plain substring keyword
    # matching is sensitive to this, so both forms need listing explicitly.
    "HAS_RATING": ["credit rating", "rated", "rating agency", "rating agencies", "credit rated"],
    "RATED_BY": ["rating agency", "rating agencies", "rated by"],
    "BELONGS_TO_SEGMENT": ["banking segment"],  # keep narrow: "segment does"/"which segment" also match unrelated "customer segment" questions
    "OFFERS_PRODUCT": ["product", "offers", "offer", "home loan", "deposit product"],
    # Had no keyword entry at all -- "wealth management"-type facts never got
    # into prop1_list on their own merit, only ever included by accident when
    # some OTHER keyword in the same query also happened to match.
    "PROVIDES_SERVICE": ["service", "services", "offer", "offers", "provides"],
    "SERVES": ["customer segment", "customers segment", "which customers"],
    # Missing entirely -- "X focuses on serving the MSME segment"/"maintains a
    # strong retail focus"/"maintains an agricultural focus" (17 edges) are
    # filed under TARGETS, a different relation from SERVES, but answer the
    # exact same "which customer segments does it serve" question.
    "TARGETS": ["customer segment", "customers segment", "which customers", "product", "service"],
    "PARTNERS_WITH": ["partnership", "partners with", "partnered with"],
    "COMPETES_WITH": ["competitor", "competes with", "competition"],
    # Also carries controlling-stake/promoter facts, not just "subsidiary of"/
    # "parent company" -- confirmed directly: RBL Bank's ONLY fact answering
    # "who is its government promoter or largest shareholder" ("Emirates NBD
    # Bank completed the acquisition of a 60% controlling stake in RBL Bank,
    # becoming its promoter under RBI norms") is filed under this relation, not
    # OwnershipStake/HELD_BY/OWNS_STAKE_IN like every other company's version of
    # the same fact -- a fourth synonymous relation name for the same concept.
    "CorporateStructureRelation": [
        "subsidiary of", "parent company", "corporate group",
        "promoter", "shareholder", "shareholding", "% held", "percent held", "held by", "stake in",
    ],
    # Only helps when nnp1 lands directly on the Metric/company node that the
    # Source is attached to -- "primary source cited for X's metrics" is really a
    # two-hop Company -> Metric -> Source traversal this single-hop keyword can't
    # fully cover, but it's a free win wherever the direct edge does exist.
    # The real relation name in the graph is "SupportedByRelation" (matching the
    # edge_types Pydantic class), not "SUPPORTED_BY" -- that wrong name never
    # matched anything, so every "primary source cited" query fell through to
    # attribute_lookup (node properties only) and failed 100% of the time (38/38
    # in the grounded benchmark) even though the fact is a simple, direct
    # Company-[SupportedByRelation]-Source edge already sitting in the graph.
    "SupportedByRelation": ["primary source", "source cited", "cited source", "sourced from"],
    # Third spelling of the same citation edge in the re-ingested graph (52 of
    # them, vs 196 SupportedByRelation and 10 SUPPORTED_BY_RELATION) -- the
    # casing-variant expansion in template_resolver covers the last one
    # automatically, but SUPPORTED_BY drops the "Relation" suffix entirely so it
    # has to be listed explicitly.
    "SUPPORTED_BY": ["primary source", "source cited", "cited source", "sourced from"],
    # Only 5/72 companies in this dataset have any overseas-presence fact at all --
    # most "not covered" answers here are a genuine source-text gap, not a missed
    # keyword, but the few that do exist are split across OPERATES_IN and
    # JOINT_VENTURE_WITH rather than a single relation name.
    "OPERATES_IN": ["overseas", "representative office", "international presence", "foreign branch"],
    "JOINT_VENTURE_WITH": ["joint venture", "joint-venture", "jv equity", "jv stake"],
    "USES_TECHNOLOGY": ["technology platform", "core banking system", "core banking"],
    # "branch" support was missing entirely -- a query like "how many branches
    # and ATMs does Indian Overseas Bank operate" instead false-matched
    # OPERATES_IN's "overseas" keyword purely because the COMPANY's own name
    # ("Indian Overseas Bank") contains that substring, not because the query
    # was actually asking about overseas presence -- and OPERATES_IN has no
    # branch-count facts at all, so prop1_list came back empty of anything real.
    "USES_CHANNEL": [
        "digital channel", "distribution channel", "banking channel",
        "branch", "branches", "branch network", "atm", "atms",
    ],
    # Also carries branch-network facts (HDFC Bank's "depends on its expansive
    # branch network to secure retail deposits" is its ONLY branch-related fact
    # in the graph, filed under this relation rather than USES_CHANNEL). Kept
    # narrowly scoped to branch/ATM terms specifically -- DEPENDS_ON is
    # otherwise a very broad, generic risk-factor relation (investor ties, RBI
    # approvals, gold prices, ...) and would pull in noise on any wider keyword.
    "DEPENDS_ON": ["branch", "branches", "branch network", "atm", "atms"],
    # "weakness"/"risk" queries had no keyword mapping at all -- prop1_list came
    # back empty, so they either fell into the unfiltered 50-row F_Simple dump or,
    # once "regulatory" pulled in REGULATED_BY, dead-ended in attribute_lookup
    # (a node-property dump) without ever trying the actual risk/weakness edges.
    # HAS_WEAKNESS and HAS_RISK_DRIVER describe the same underlying concept
    # (a company's flagged weak point) under different relation names depending
    # on which one ingestion happened to use for that company -- both must be
    # candidates for either phrasing or the answer is missed for whichever name
    # that specific company's fact wasn't stored under.
    # Same metric-vocabulary merge as ExposureRelation above, for the same
    # reason: a specific figure the query asks about by name (CET1, NIM, gross
    # NPA...) routinely turns out to be filed under one of these "narrative"
    # relations instead of HAS_METRIC/MetricObservation, because that's simply
    # how the sentence carrying the number happened to be framed at ingestion
    # time ("a weakness is its CET1 capital of 13.33%", "a strength is 19%
    # advances growth", "expects margin growth from an improving CASA mix").
    # Confirmed directly: IDFC FIRST Bank's ONLY CET1 fact in the whole graph
    # lives on a HAS_WEAKNESS edge -- "what's the CET1 ratio" queries had zero
    # chance of finding it via HAS_METRIC/MetricObservation keywords alone.
    "HAS_WEAKNESS": ["weakness", "vulnerability", "vulnerable", "risk", "risks", "exposed to", "exposure to", *_METRIC_TERMS],
    "HAS_RISK_DRIVER": ["weakness", "vulnerability", "vulnerable", "risk", "risks", "exposed to", "exposure to", *_METRIC_TERMS],
    # EXPOSED_TO was removed earlier in favour of ExposureRelation, because the
    # then-current graph had zero EXPOSED_TO edges -- but the re-ingested graph
    # has 56 of them, alongside the ExposureRelation ones. Which of the two an
    # episode's facts land under is not stable across ingestion runs, so both
    # need mapping. Same story for the other exposure-flavoured names below.
    "EXPOSED_TO": [
        "weakness", "vulnerability", "vulnerable", "risk", "risks", "exposed to", "exposure to",
    ],
    "HAS_LOAN_EXPOSURE_TO": ["advances", "loan book", "loan", "loans", "exposed to", "exposure to"],
    "HAS_STRATEGIC_EXPOSURE_TO": ["exposed to", "exposure to", "risk", "risks"],
    "INCREASES_EXPOSURE_TO": ["exposed to", "exposure to", "risk", "risks"],
    "THREATENS": ["threat", "threatens", "threatened by"],
    "SUBJECT_TO": ["subject to"],
    # Not in the map at all -- Bank of Baroda's flagged weakness (a ₹5,680 crore
    # legal settlement tied to NMC Health) is filed under this relation, a fifth
    # distinct name (alongside HAS_WEAKNESS/HAS_RISK_DRIVER/ExposureRelation/
    # THREATENS) for the same "what's wrong with this company" query family.
    "INVOLVED_IN_LITIGATION_WITH": [
        "weakness", "vulnerability", "vulnerable", "risk", "risks", "exposed to", "exposure to",
        "litigation", "lawsuit", "legal", "settlement",
    ],
    "HAS_STRENGTH": ["strength", "strong point", "advantage", "competitive advantage", *_METRIC_TERMS],
    # "digital"/"technology"/"platform"/"channel" merged in too: a "technology
    # platform / digital channel" question can be answered by a growth-driver
    # fact ("Indian Bank anticipates growth driven by continued digital-
    # transaction growth") that USES_TECHNOLOGY/USES_CHANNEL's own keywords
    # never reach, since neither the growth-driver fact nor the query
    # necessarily says "core banking"/"technology platform" verbatim.
    "HAS_GROWTH_DRIVER": [
        "growth driver", "digital", "technology", "platform", "channel", "core banking",
        *_METRIC_TERMS,
    ],
    "HAS_OPPORTUNITY": ["opportunity", "opportunities", *_METRIC_TERMS],
    "HAS_MARGIN_DRIVER": ["margin driver", *_METRIC_TERMS],
    # Same driver family, two more names the re-ingested graph uses (52 and 36
    # edges) -- a "what drives X's return on equity / valuation" or plain
    # metric question should reach these the same way it reaches the margin and
    # growth drivers above.
    "HAS_ROE_DRIVER": ["roe driver", "return on equity", "roe", *_METRIC_TERMS],
    "HAS_VALUATION_DRIVER": ["valuation driver", "valuation", *_METRIC_TERMS],
    # Cleanly loan/product-book focused ("IDFC FIRST Bank has heavily focused on
    # MSME lending", "retail loan portfolio", "rural loan-book") -- unlike
    # DEPENDS_ON (also seen carrying loan-adjacent facts, but overwhelmingly a
    # generic risk-factor relation covering investor relationships, RBI
    # approvals, gold prices, etc.), this one is narrow enough to map safely.
    "HAS_ASSET_EXPOSURE_TO": ["product", "service", "loan", "loans", "offer"],
    # "competitive advantage" and "key strength" are used interchangeably in this
    # dataset for the same underlying fact (Indian Overseas Bank's only match for
    # either phrasing -- "its clearest strength is its improving asset quality" --
    # is filed under HAS_STRENGTH, not HAS_ADVANTAGE), so each keyword list now
    # covers both relations.
    "HAS_ADVANTAGE": ["competitive advantage", "advantage", "strength", "strong point"],
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
    "F_GlobalRank": {"COMPANY", "METRIC"},
    "F_GroupAggregate": {"COMPANY", "METRIC"},
    "F_CompToGroupAverage": {"COMPANY", "METRIC"},
}

# Canonical BankingSegment node name -> the query-text synonyms that refer to it.
# Matched directly against raw query text (not through NER/gazetteer) since the
# segment is a query-level filter/grouping key, not a "mentioned entity" in the
# usual sense -- "Public Sector Banks" rarely appears in a query exactly as
# stored on the node.
BANKING_SEGMENT_KEYWORDS: dict[str, list[str]] = {
    "Public Sector Banks": ["public sector bank", "psu bank", "psb"],
    "Private Sector Banks": ["private sector bank"],
    "Small Finance Banks": ["small finance bank"],
}

# Fallback company -> canonical BankingSegment classification, used ONLY when
# the graph has no BankingSegment edge (or an inconsistent one -- "universal
# bank" is a real, separate operating-model tag some episodes used instead of
# the PSU/Private/SFB ownership classification) for that company. Confirmed by
# direct inspection: of the ~39 real listed banks in this dataset, the large
# majority are either completely untagged (Axis Bank, Bank of India, Union
# Bank of India, IndusInd Bank, Kotak Mahindra Bank, Yes Bank, IDFC FIRST
# Bank, ...) or tagged only with the unrelated "universal bank"/"A Class
# Scheduled Commercial Bank" operating-model labels (SBI, PNB, Indian
# Overseas Bank, UCO Bank, Karnataka Bank) instead of an ownership-category
# segment -- this alone was the largest remaining blocker on cross-company
# segment-ranking/averaging queries (F_GlobalRank's segment-filtered mode,
# F_GroupAggregate, F_CompToGroupAverage). RBI's PSU/Private/SFB categories
# are stable public regulatory classifications, not a judgment call, so a
# static table is safe here -- unlike, say, guessing at a financial figure.
# IDBI Bank deliberately excluded: its classification is genuinely contested
# (LIC-majority-owned former PSU, RBI-reclassified as private in 2019) and the
# graph already carries both tags for it, which is left as-is rather than
# overridden.
COMPANY_SEGMENT_FALLBACK: dict[str, str] = {
    # Public Sector Banks (nationalized / government-majority-owned)
    "State Bank of India": "Public Sector Banks",
    "Punjab National Bank": "Public Sector Banks",
    "Bank of Baroda": "Public Sector Banks",
    "Canara Bank": "Public Sector Banks",
    "Union Bank of India": "Public Sector Banks",
    "Bank of India": "Public Sector Banks",
    "Indian Bank": "Public Sector Banks",
    "Central Bank of India": "Public Sector Banks",
    "Indian Overseas Bank": "Public Sector Banks",
    "UCO Bank": "Public Sector Banks",
    "Bank of Maharashtra": "Public Sector Banks",
    "Punjab & Sind Bank": "Public Sector Banks",
    # Private Sector Banks
    "HDFC Bank": "Private Sector Banks",
    "ICICI Bank": "Private Sector Banks",
    "Axis Bank": "Private Sector Banks",
    "Kotak Mahindra Bank": "Private Sector Banks",
    "IndusInd Bank": "Private Sector Banks",
    "Yes Bank": "Private Sector Banks",
    "IDFC FIRST Bank": "Private Sector Banks",
    "Federal Bank": "Private Sector Banks",
    "RBL Bank": "Private Sector Banks",
    "South Indian Bank": "Private Sector Banks",
    "Karur Vysya Bank": "Private Sector Banks",
    "City Union Bank": "Private Sector Banks",
    "DCB Bank": "Private Sector Banks",
    "Karnataka Bank": "Private Sector Banks",
    "Tamilnad Mercantile Bank": "Private Sector Banks",
    "Bandhan Bank": "Private Sector Banks",
    "Dhanlaxmi Bank": "Private Sector Banks",
    "Jammu & Kashmir Bank": "Private Sector Banks",
    "CSB Bank": "Private Sector Banks",
    # Small Finance Banks
    "AU Small Finance Bank": "Small Finance Banks",
    "Equitas Small Finance Bank": "Small Finance Banks",
    "ESAF Small Finance Bank": "Small Finance Banks",
    "Ujjivan Small Finance Bank": "Small Finance Banks",
    "Suryoday Small Finance Bank": "Small Finance Banks",
    "Utkarsh Small Finance Bank": "Small Finance Banks",
    "Jana Small Finance Bank": "Small Finance Banks",
}
