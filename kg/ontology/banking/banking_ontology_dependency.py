from kg.ontology.banking.banking_ontology_models import (
    # Layer 1 — structural entities
    Sector, Industry, BankingSegment, Company, CorporateGroup, Product, Service,
    CustomerSegment, Geography, DistributionChannel, Regulator, License,
    # Layer 2 — business and economic entities
    BusinessModel, OperatingModel, RevenueStream, FundingSource, AssetClass,
    LiabilityClass, LoanSegment, DepositSegment, InvestmentClass, Capability,
    Technology, CompetitiveArena, Partnership, Ecosystem, Metric,
    # Layer 3 — analyst entities
    Strength, Weakness, Opportunity, Threat, Risk, Dependency, Driver,
    CompetitiveAdvantage, StrategicPosition, Claim, Condition, Outcome,
    Evidence, Source,
    # Governance / ownership / ratings extension
    Person, PromoterGroup, Shareholder, CreditRatingAgency, CreditRating,
    MacroeconomicFactor,
    # Edge types that carry their own properties
    CorporateStructureRelation, FundingProfile, ExposureRelation,
    MetricObservation, CausalRelationship, RegulatoryRelation,
    SupportedByRelation, Relationship, GovernanceRole, OwnershipStake,
)

# ============================================================
# ENTITY TYPES
# All node classes from layers 1-3. Claim, Condition, and Outcome
# are entities too (per section 31's node-vs-property rule) —
# Claim's subject_entity_id / supporting_entity_ids are properties
# on the node itself, not separate graph edges, so Claim has no
# corresponding entry in edge_type_map below.
# ============================================================

entity_types = {
    "Sector": Sector,
    "Industry": Industry,
    "BankingSegment": BankingSegment,
    "Company": Company,
    "CorporateGroup": CorporateGroup,
    "Product": Product,
    "Service": Service,
    "CustomerSegment": CustomerSegment,
    "Geography": Geography,
    "DistributionChannel": DistributionChannel,
    "Regulator": Regulator,
    "License": License,
    "BusinessModel": BusinessModel,
    "OperatingModel": OperatingModel,
    "RevenueStream": RevenueStream,
    "FundingSource": FundingSource,
    "AssetClass": AssetClass,
    "LiabilityClass": LiabilityClass,
    "LoanSegment": LoanSegment,
    "DepositSegment": DepositSegment,
    "InvestmentClass": InvestmentClass,
    "Capability": Capability,
    "Technology": Technology,
    "CompetitiveArena": CompetitiveArena,
    "Partnership": Partnership,
    "Ecosystem": Ecosystem,
    "Metric": Metric,
    "Strength": Strength,
    "Weakness": Weakness,
    "Opportunity": Opportunity,
    "Threat": Threat,
    "Risk": Risk,
    "Dependency": Dependency,
    "Driver": Driver,
    "CompetitiveAdvantage": CompetitiveAdvantage,
    "StrategicPosition": StrategicPosition,
    "Claim": Claim,
    "Condition": Condition,
    "Outcome": Outcome,
    "Evidence": Evidence,
    "Source": Source,
    # Governance / ownership / ratings extension
    "Person": Person,
    "PromoterGroup": PromoterGroup,
    "Shareholder": Shareholder,
    "CreditRatingAgency": CreditRatingAgency,
    "CreditRating": CreditRating,
    "MacroeconomicFactor": MacroeconomicFactor,
}

# ============================================================
# EDGE TYPES
# Relationships with their own properties keep their dedicated
# class. Every plain, property-free relationship name from
# RELATIONSHIP_TYPES maps to the generic `Relationship` model
# (a thin wrapper carrying only relation_type) so every string
# used in edge_type_map below resolves to a real class here.
# ============================================================

edge_types = {
    # Property-carrying edges
    "CorporateStructureRelation": CorporateStructureRelation,
    "FundingProfile": FundingProfile,
    "ExposureRelation": ExposureRelation,
    "MetricObservation": MetricObservation,
    "CausalRelationship": CausalRelationship,
    "RegulatoryRelation": RegulatoryRelation,
    "SupportedByRelation": SupportedByRelation,
    "GovernanceRole": GovernanceRole,
    "OwnershipStake": OwnershipStake,
    # Plain directional relationships (no extra properties)
    "BELONGS_TO": Relationship,
    "BELONGS_TO_SEGMENT": Relationship,
    "OPERATES_IN_INDUSTRY": Relationship,
    "HAS_COMPANY": Relationship,
    "OFFERS_PRODUCT": Relationship,
    "TARGETS": Relationship,
    "DISTRIBUTED_THROUGH": Relationship,
    "PROVIDES_SERVICE": Relationship,
    "SERVES": Relationship,
    "FUNDED_BY": Relationship,
    "HAS_REVENUE_STREAM": Relationship,
    "GENERATES_REVENUE_FROM": Relationship,
    "USES_CHANNEL": Relationship,
    "OPERATES_IN": Relationship,
    "HAS_STRONG_PRESENCE_IN": Relationship,
    "HAS_STRATEGIC_EXPOSURE_TO": Relationship,
    "HAS_BUSINESS_MODEL": Relationship,
    "HAS_OPERATING_MODEL": Relationship,
    "HAS_CAPABILITY": Relationship,
    "ENABLES": Relationship,
    "CREATES_ADVANTAGE_IN": Relationship,
    "USES_TECHNOLOGY": Relationship,
    "COMPETES_WITH": Relationship,
    "COMPETES_IN": Relationship,
    "HAS_ADVANTAGE_IN": Relationship,
    "DIFFERENTIATES_FROM": Relationship,
    "HAS_ADVANTAGE": Relationship,
    "IMPROVES": Relationship,
    "HAS_STRENGTH": Relationship,
    "SUPPORTS": Relationship,
    "HAS_WEAKNESS": Relationship,
    "INCREASES_EXPOSURE_TO": Relationship,
    "HAS_OPPORTUNITY": Relationship,
    "DEPENDS_ON": Relationship,
    "THREATENS": Relationship,
    "AFFECTS": Relationship,
    "IMPACTS": Relationship,
    "MITIGATED_BY": Relationship,
    "CREATES_RISK": Relationship,
    "CONSTRAINS": Relationship,
    "HAS_METRIC": Relationship,
    "MEASURES": Relationship,
    "HAS_DRIVER": Relationship,
    "HAS_GROWTH_DRIVER": Relationship,
    "HAS_MARGIN_DRIVER": Relationship,
    "HAS_ROE_DRIVER": Relationship,
    "HAS_RISK_DRIVER": Relationship,
    "HAS_VALUATION_DRIVER": Relationship,
    "PARTNERS_WITH": Relationship,
    "PART_OF_ECOSYSTEM": Relationship,
    "POSITIONED_AS": Relationship,
    "FROM": Relationship,
    "INFLUENCES": Relationship,
    "MEMBER_OF": Relationship,
    "RATED_BY": Relationship,
    "HAS_RATING": Relationship,
}

# ============================================================
# EDGE TYPE MAP
# ("SourceEntity", "TargetEntity") -> [applicable edge type keys]
#
# "Entity" is a wildcard meaning "any entity type" — used for the
# ontology's generic causal connectors (ENABLES, SUPPORTS,
# DEPENDS_ON, CONSTRAINS, IMPROVES, CausalRelationship itself),
# which the RTF explicitly allows between any two nodes that
# participate in a causal chain (section 23).
# ============================================================

edge_type_map = {
    # Sector / industry / segment hierarchy
    ("Industry", "Sector"): ["BELONGS_TO"],
    ("Company", "Industry"): ["OPERATES_IN_INDUSTRY"],
    ("Company", "BankingSegment"): ["BELONGS_TO_SEGMENT"],
    ("Industry", "Company"): ["HAS_COMPANY"],

    # Corporate structure — CorporateStructureRelation.relation_type carries
    # which of PART_OF_GROUP / PARENT_OF / SUBSIDIARY_OF / ASSOCIATE_OF /
    # AFFILIATED_WITH / JOINT_VENTURE_WITH applies
    ("Company", "CorporateGroup"): ["CorporateStructureRelation"],
    ("Company", "Company"): [
        "CorporateStructureRelation", "COMPETES_WITH", "DIFFERENTIATES_FROM",
    ],

    # Products, services, customers, channels
    ("Company", "Product"): ["OFFERS_PRODUCT"],
    ("Product", "CustomerSegment"): ["TARGETS"],
    ("Product", "DistributionChannel"): ["DISTRIBUTED_THROUGH"],
    ("Company", "Service"): ["PROVIDES_SERVICE"],
    ("Service", "CustomerSegment"): ["SERVES"],
    ("Company", "CustomerSegment"): ["SERVES", "TARGETS"],
    ("Company", "DistributionChannel"): ["USES_CHANNEL"],

    # Funding, revenue, assets, loans
    ("Company", "FundingSource"): ["FUNDED_BY", "FundingProfile"],
    ("Company", "RevenueStream"): ["HAS_REVENUE_STREAM", "GENERATES_REVENUE_FROM"],
    # ExposureRelation.relation_type carries HAS_ASSET_EXPOSURE_TO / HAS_LOAN_EXPOSURE_TO / EXPOSED_TO
    ("Company", "AssetClass"): ["ExposureRelation"],
    ("Company", "LoanSegment"): ["ExposureRelation"],

    # Geography
    ("Company", "Geography"): [
        "OPERATES_IN", "HAS_STRONG_PRESENCE_IN", "HAS_STRATEGIC_EXPOSURE_TO",
    ],
    ("Geography", "Geography"): ["BELONGS_TO"],

    # Business / operating model
    ("Company", "BusinessModel"): ["HAS_BUSINESS_MODEL"],
    ("Company", "OperatingModel"): ["HAS_OPERATING_MODEL"],

    # Capability, technology, competition, advantage
    ("Company", "Capability"): ["HAS_CAPABILITY"],
    ("Technology", "Capability"): ["ENABLES"],
    ("Company", "Technology"): ["USES_TECHNOLOGY"],
    ("Company", "CompetitiveArena"): ["COMPETES_IN"],
    ("Capability", "CompetitiveArena"): ["CREATES_ADVANTAGE_IN"],
    ("Company", "CompetitiveAdvantage"): ["HAS_ADVANTAGE"],
    ("CompetitiveAdvantage", "CompetitiveArena"): ["HAS_ADVANTAGE_IN", "CREATES_ADVANTAGE_IN"],

    # SWOT
    ("Company", "Strength"): ["HAS_STRENGTH"],
    ("Company", "Weakness"): ["HAS_WEAKNESS"],
    ("Weakness", "Risk"): ["INCREASES_EXPOSURE_TO"],
    ("Company", "Opportunity"): ["HAS_OPPORTUNITY"],
    ("Company", "Threat"): ["THREATENS"],
    ("Threat", "Company"): ["THREATENS", "AFFECTS"],

    # Risk and dependency
    ("Company", "Risk"): ["ExposureRelation"],
    ("Risk", "Company"): ["AFFECTS", "IMPACTS"],
    ("Risk", "Capability"): ["MITIGATED_BY"],
    ("Company", "Dependency"): ["DEPENDS_ON"],
    ("Dependency", "Risk"): ["CREATES_RISK"],

    # Metrics
    ("Company", "Metric"): ["HAS_METRIC", "MetricObservation"],
    ("Metric", "Company"): ["MEASURES"],

    # Drivers
    ("Company", "Driver"): [
        "HAS_DRIVER", "HAS_GROWTH_DRIVER", "HAS_MARGIN_DRIVER",
        "HAS_ROE_DRIVER", "HAS_RISK_DRIVER", "HAS_VALUATION_DRIVER",
    ],

    # Regulation — RegulatoryRelation.relation_type carries REGULATED_BY /
    # HAS_LICENSE / SUBJECT_TO / REQUIRES
    ("Company", "Regulator"): ["RegulatoryRelation"],
    ("Company", "License"): ["RegulatoryRelation"],

    # Partnerships and ecosystem
    ("Company", "Partnership"): ["PARTNERS_WITH"],
    ("Company", "Ecosystem"): ["PART_OF_ECOSYSTEM"],

    # Strategic positioning
    ("Company", "StrategicPosition"): ["POSITIONED_AS"],

    # Governance and people (extension)
    ("Person", "Company"): ["GovernanceRole"],
    ("Person", "PromoterGroup"): ["MEMBER_OF"],

    # Ownership (extension)
    ("Shareholder", "Company"): ["OwnershipStake"],
    ("PromoterGroup", "Company"): ["OwnershipStake"],

    # Credit ratings (extension)
    ("Company", "CreditRatingAgency"): ["RATED_BY"],
    ("Company", "CreditRating"): ["HAS_RATING"],
    ("CreditRatingAgency", "CreditRating"): ["HAS_RATING"],

    # Macroeconomic drivers (extension) — external factors feeding into
    # the same causal chains as internal drivers (RTF section 23)
    ("MacroeconomicFactor", "Driver"): ["CausalRelationship"],
    ("MacroeconomicFactor", "Company"): ["CausalRelationship"],

    # Provenance
    ("Entity", "Evidence"): ["SupportedByRelation"],
    ("Evidence", "Source"): ["FROM"],

    # Generic causal connectors — apply broadly across the graph,
    # not tied to a specific entity pair (RTF section 23)
    ("Entity", "Entity"): [
        "CausalRelationship", "ENABLES", "SUPPORTS", "IMPROVES",
        "CONSTRAINS", "DEPENDS_ON", "INFLUENCES",
    ],
}