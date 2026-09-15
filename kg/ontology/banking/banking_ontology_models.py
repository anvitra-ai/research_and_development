from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


# ============================================================
# LAYER 1 — STRUCTURAL ENTITY TYPES
# ============================================================

class Sector(BaseModel):
    """Top-level sector classification (e.g. Financial Services)."""
    name: Optional[str] = Field(None, description="Sector name")
    parent_sector: Optional[str] = Field(None, description="Parent sector, if nested")


class Industry(BaseModel):
    """Industry within a sector (e.g. Banking)."""
    name: Optional[str] = Field(None, description="Industry name")
    sector: Optional[str] = Field(None, description="Parent sector name")


class BankingSegment(BaseModel):
    """Banking segment classification, e.g. Public Sector Bank, Private Sector Bank."""
    name: Optional[str] = Field(None, description="Segment name (e.g. Private Sector Banks)")
    description: Optional[str] = Field(None, description="Segment description")


class Company(BaseModel):
    """A listed company (bank or banking-adjacent institution)."""
    legal_name: Optional[str] = Field(None, description="Registered legal name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    isin: Optional[str] = Field(None, description="ISIN identifier")
    listed: Optional[bool] = Field(None, description="Whether the company is publicly listed")
    exchange: Optional[str] = Field(None, description="Primary listing exchange")
    country: Optional[str] = Field(None, description="Country of incorporation")
    sector: Optional[str] = Field(None, description="Sector name")
    industry: Optional[str] = Field(None, description="Industry name")
    banking_segment: Optional[str] = Field(None, description="Banking segment name")
    website: Optional[str] = Field(None, description="Company website")
    description: Optional[str] = Field(None, description="Free-text company description")


class CorporateGroup(BaseModel):
    """A corporate group / conglomerate that companies may belong to."""
    name: Optional[str] = Field(None, description="Group name")
    description: Optional[str] = Field(None, description="Group description")


class Product(BaseModel):
    """A banking product offered to customers."""
    name: Optional[str] = Field(None, description="Product name (e.g. Home Loan)")
    category: Optional[str] = Field(
        None,
        description="Deposit, Loan, Card, Payment, Wealth, Insurance, Forex, Treasury, or Investment Product",
    )
    description: Optional[str] = Field(None, description="Product description")


class Service(BaseModel):
    """A banking service (non-product offering)."""
    name: Optional[str] = Field(None, description="Service name (e.g. Trade Finance)")
    description: Optional[str] = Field(None, description="Service description")


class CustomerSegment(BaseModel):
    """A customer segment served by a company."""
    name: Optional[str] = Field(
        None,
        description=(
            "Retail, Mass Market, Affluent, HNI, UHNI, SME, Mid Market, Large Corporate, "
            "Government, Financial Institution, Agriculture, or Rural Customer"
        ),
    )
    description: Optional[str] = Field(None, description="Segment description")


class Geography(BaseModel):
    """A geographic entity: country, state, region, or city."""
    name: Optional[str] = Field(None, description="Geography name")
    level: Optional[str] = Field(None, description="Country, State, Region, or City")
    area_type: Optional[str] = Field(None, description="Urban, Semi Urban, Rural, or International Market")
    parent_geography: Optional[str] = Field(None, description="Parent geography name, if nested")


class DistributionChannel(BaseModel):
    """A channel through which products/services reach customers."""
    name: Optional[str] = Field(
        None,
        description=(
            "Branch, ATM, Mobile Application, Web Platform, API, Relationship Manager, "
            "Direct Sales, Agent Partner, Merchant Network, or Corporate Partnership"
        ),
    )


class Regulator(BaseModel):
    """A regulatory body."""
    name: Optional[str] = Field(None, description="Regulator name (e.g. RBI, SEBI)")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction covered")


class License(BaseModel):
    """A license or authorization held by a company."""
    name: Optional[str] = Field(None, description="License name/type")
    issuing_regulator: Optional[str] = Field(None, description="Regulator that issued the license")
    status: Optional[str] = Field(None, description="Current status of the license")


# ============================================================
# LAYER 2 — BUSINESS AND ECONOMIC ENTITY TYPES
# ============================================================

class BusinessModel(BaseModel):
    """A business model dimension for a company."""
    dimension: Optional[str] = Field(
        None,
        description=(
            "Customer Model, Revenue Model, Funding Model, Distribution Model, "
            "Operating Model, Risk Model, or Competitive Model"
        ),
    )
    description: Optional[str] = Field(None, description="Description of this business model dimension")


class OperatingModel(BaseModel):
    """A company's operating model archetype."""
    name: Optional[str] = Field(
        None,
        description="Branch Led, Digital First, Hybrid, Relationship Led, Partnership Led, or Platform Led",
    )


class RevenueStream(BaseModel):
    """A source of revenue for a company."""
    name: Optional[str] = Field(
        None,
        description=(
            "Interest Income, Fee Income, Commission Income, Treasury Income, Forex Income, "
            "Payment Income, Wealth Income, Insurance Distribution Income, or Other Operating Income"
        ),
    )


class FundingSource(BaseModel):
    """A source of funding/liabilities for a company."""
    name: Optional[str] = Field(
        None,
        description=(
            "CASA, Savings Deposits, Current Deposits, Term Deposits, Institutional Deposits, "
            "Borrowings, Bonds, or Interbank Funding"
        ),
    )


class AssetClass(BaseModel):
    """A category of asset held by a company."""
    name: Optional[str] = Field(
        None, description="Loan Portfolio, Investments, Cash, Interbank Assets, or Other Assets"
    )


class LiabilityClass(BaseModel):
    """A category of liability held by a company."""
    name: Optional[str] = Field(None, description="Liability class name")


class LoanSegment(BaseModel):
    """A loan portfolio segment."""
    name: Optional[str] = Field(
        None,
        description=(
            "Retail Lending (Home/Vehicle/Personal/Credit Card/Education Loan) or Corporate "
            "Lending (Large/Mid Corporate, SME, Agriculture, Infrastructure Lending)"
        ),
    )
    segment_group: Optional[str] = Field(None, description="Retail Lending or Corporate Lending")


class DepositSegment(BaseModel):
    """A deposit portfolio segment."""
    name: Optional[str] = Field(None, description="Deposit segment name")


class InvestmentClass(BaseModel):
    """A class of investment holding."""
    name: Optional[str] = Field(None, description="Investment class name")


class Capability(BaseModel):
    """An operational capability a company possesses."""
    name: Optional[str] = Field(
        None,
        description=(
            "Underwriting, Risk Management, Fraud Detection, Customer Acquisition, Customer "
            "Retention, Cross Selling, Collections, Data Analytics, Relationship Management, "
            "Treasury Management, Digital Onboarding, or Payment Processing"
        ),
    )


class Technology(BaseModel):
    """A technology category used by a company."""
    name: Optional[str] = Field(
        None,
        description=(
            "Core Banking, Cloud, Artificial Intelligence, Data Platform, Cybersecurity, "
            "Mobile Banking, API, Payment Infrastructure, or Automation"
        ),
    )


class CompetitiveArena(BaseModel):
    """A market/arena in which companies compete."""
    name: Optional[str] = Field(
        None,
        description=(
            "Deposits, Retail Lending, Corporate Lending, SME Lending, Credit Cards, Payments, "
            "Wealth Management, Digital Banking, or Transaction Banking"
        ),
    )


class Partnership(BaseModel):
    """A persistent partnership entity (not a one-off partnership event)."""
    name: Optional[str] = Field(None, description="Partnership name/description")
    category: Optional[str] = Field(
        None, description="Technology, Distribution, Lending, Payments, Insurance, Wealth, or Data"
    )


class Ecosystem(BaseModel):
    """An ecosystem a company participates in."""
    name: Optional[str] = Field(
        None,
        description=(
            "Banking, Insurance, Asset Management, Broking, Payments, Lending, Wealth, "
            "Consumer, or Platform"
        ),
    )


class Metric(BaseModel):
    """A reusable metric definition (company-specific values live in MetricObservation)."""
    name: Optional[str] = Field(None, description="Metric name, e.g. ROA, ROE, NIM, GNPA, CET1")
    category: Optional[str] = Field(
        None, description="Growth, Profitability, Efficiency, Asset Quality, or Capital"
    )
    unit: Optional[str] = Field(None, description="Default unit of measurement, e.g. percent, ratio, INR crore")


# ============================================================
# LAYER 1 EXTENSION — GOVERNANCE, OWNERSHIP, RATINGS
# Added to close gaps identified against the original RTF:
# no way to represent who runs or owns a company, or how
# rating agencies assess it.
# ============================================================

class Person(BaseModel):
    """An individual associated with a company's governance or management."""
    name: Optional[str] = Field(None, description="Full name")
    role_category: Optional[str] = Field(None, description="Executive, Board Member, or Promoter")


class PromoterGroup(BaseModel):
    """A promoter entity (individual or family group) holding a controlling stake."""
    name: Optional[str] = Field(None, description="Promoter group name")
    description: Optional[str] = Field(None, description="Description of the promoter group")


class Shareholder(BaseModel):
    """A category of institutional shareholder holding equity in a company."""
    name: Optional[str] = Field(None, description="Shareholder name, if a specific institution")
    holder_type: Optional[str] = Field(
        None, description="Promoter, FII, DII, Mutual Fund, Insurance, or Retail"
    )


class CreditRatingAgency(BaseModel):
    """A credit rating agency, e.g. CRISIL, ICRA, Moody's, S&P."""
    name: Optional[str] = Field(None, description="Rating agency name")


class CreditRating(BaseModel):
    """A specific credit rating assigned to a company or capital instrument."""
    scale_value: Optional[str] = Field(None, description="Rating symbol, e.g. AAA, AA+, Baa1")
    outlook: Optional[str] = Field(None, description="Stable, Positive, Negative, or Under Review")
    rated_entity_type: Optional[str] = Field(None, description="Issuer rating or Instrument rating")
    period: Optional[str] = Field(None, description="Date/period the rating reflects")


class MacroeconomicFactor(BaseModel):
    """An external macroeconomic or policy factor relevant to bank performance."""
    name: Optional[str] = Field(
        None,
        description="e.g. Repo Rate, GDP Growth, Inflation, Credit Cycle Phase, Systemic Liquidity",
    )
    category: Optional[str] = Field(
        None, description="Monetary Policy, Growth, Inflation, Liquidity, or Credit Cycle"
    )


# ============================================================
# LAYER 3 — ANALYST ENTITY TYPES
# ============================================================

class Strength(BaseModel):
    """An analytical strength attributed to a company."""
    category: Optional[str] = Field(
        None,
        description=(
            "Financial, Funding, Distribution, Brand, Technology, Capability, Customer, "
            "Scale, Management, or Strategic"
        ),
    )
    statement: Optional[str] = Field(None, description="Description of the strength")


class Weakness(BaseModel):
    """An analytical weakness attributed to a company."""
    category: Optional[str] = Field(
        None,
        description=(
            "Financial, Funding, Distribution, Technology, Customer Concentration, "
            "Geographic Concentration, Product Concentration, Scale, or Capability"
        ),
    )
    statement: Optional[str] = Field(None, description="Description of the weakness")


class Opportunity(BaseModel):
    """An analytical opportunity attributed to a company."""
    category: Optional[str] = Field(
        None,
        description=(
            "Market/Customer/Product/Geographic/Digital/Margin Expansion, Cross Sell, "
            "Partnership, Ecosystem, or Regulatory Opportunity"
        ),
    )
    statement: Optional[str] = Field(None, description="Description of the opportunity")


class Threat(BaseModel):
    """An analytical threat attributed to a company."""
    category: Optional[str] = Field(
        None,
        description=(
            "Competition, Technology Disruption, Regulatory Pressure, Margin Compression, "
            "Credit Deterioration, Funding Pressure, Cybersecurity, FinTech Disruption, "
            "Customer Migration, or Macroeconomic Exposure"
        ),
    )
    statement: Optional[str] = Field(None, description="Description of the threat")


class Risk(BaseModel):
    """A risk category a company is exposed to."""
    category: Optional[str] = Field(
        None,
        description=(
            "Credit, Market, Liquidity, Interest Rate, Operational, Cyber, Regulatory, "
            "Concentration, Reputation, Technology, or Funding Risk"
        ),
    )
    description: Optional[str] = Field(None, description="Risk description")


class Dependency(BaseModel):
    """A structural dependency a company has."""
    category: Optional[str] = Field(
        None,
        description=(
            "Technology, Funding, Partner, Customer, Vendor, Geographic, Distribution, "
            "or Regulatory Dependency"
        ),
    )
    description: Optional[str] = Field(None, description="Dependency description")


class Driver(BaseModel):
    """A causal driver of growth, revenue, margin, profitability, ROE, risk, competitiveness, or valuation."""
    driver_type: Optional[str] = Field(
        None,
        description=(
            "Growth, Revenue, Margin, Profitability, ROE, Risk, Competitive, Structural, "
            "or Valuation Driver"
        ),
    )
    statement: Optional[str] = Field(None, description="Description of the driver")


class CompetitiveAdvantage(BaseModel):
    """A source of competitive advantage."""
    source: Optional[str] = Field(
        None,
        description=(
            "Brand, Distribution, Scale, Cost of Funds, Customer Base, Data, Technology, "
            "Ecosystem, Regulatory License, Network Effects, or Switching Costs"
        ),
    )
    description: Optional[str] = Field(None, description="Description of the advantage")


class StrategicPosition(BaseModel):
    """A company's strategic positioning."""
    name: Optional[str] = Field(
        None,
        description=(
            "Premium, Mass Market, Digital First, Corporate Focus, Retail Focus, Rural Focus, "
            "SME Focus, Universal Bank, or Niche Bank"
        ),
    )


class Claim(BaseModel):
    """An analytical conclusion, kept structurally separate from objective facts."""
    claim_id: Optional[str] = Field(None, description="Stable, unique claim identifier")
    claim_type: Optional[str] = Field(
        None,
        description=(
            "Strength, Weakness, Opportunity, Threat, Risk Assessment, Competitive Position, "
            "Franchise Quality, Growth Driver, Margin Driver, ROE Driver, or Valuation Driver"
        ),
    )
    statement: Optional[str] = Field(None, description="The analytical statement being made")
    subject_entity_id: Optional[str] = Field(None, description="Entity the claim is about")
    supporting_entity_ids: Optional[List[str]] = Field(
        None, description="Entities cited in support of the claim"
    )
    confidence: Optional[str] = Field(None, description="High, Medium, or Low")
    evidence_strength: Optional[str] = Field(
        None, description="Direct, Strong Inference, Moderate Inference, or Weak Inference"
    )
    rationale: Optional[str] = Field(None, description="Reasoning behind the claim")


class Condition(BaseModel):
    """A precondition referenced within analyst reasoning (e.g. a causal chain)."""
    description: Optional[str] = Field(None, description="Condition description")


class Outcome(BaseModel):
    """An analytical outcome referenced within analyst reasoning."""
    description: Optional[str] = Field(None, description="Outcome description")


class Evidence(BaseModel):
    """Provenance record supporting an entity, relationship, metric, or claim."""
    evidence_id: Optional[str] = Field(None, description="Stable, unique evidence identifier")
    entity_or_claim_id: Optional[str] = Field(None, description="ID of the entity/claim this supports")
    evidence_type: Optional[str] = Field(None, description="Type of evidence, e.g. filing, disclosure, report")
    source_name: Optional[str] = Field(None, description="Name of the source")
    source_reference: Optional[str] = Field(None, description="Reference/citation for the source")
    description: Optional[str] = Field(None, description="Description of what the evidence shows")
    confidence: Optional[str] = Field(None, description="Confidence in the evidence itself")


class Source(BaseModel):
    """A named source referenced by Evidence records."""
    name: Optional[str] = Field(None, description="Source name")
    source_type: Optional[str] = Field(None, description="Regulatory filing, annual report, press release, etc.")


# ============================================================
# EDGE TYPES — relationships that carry their own properties.
# Plain directional relationships with no extra data use the
# generic Relationship model below, per the ontology's own
# node-vs-property rule (section 31) applied one level down,
# to edges rather than entities.
# ============================================================

class Relationship(BaseModel):
    """A plain directional relationship with no additional properties."""
    relation_type: Optional[str] = Field(None, description="One of RELATIONSHIP_TYPES")


class CorporateStructureRelation(BaseModel):
    """PART_OF_GROUP / PARENT_OF / SUBSIDIARY_OF / ASSOCIATE_OF / AFFILIATED_WITH / JOINT_VENTURE_WITH."""
    relation_type: Optional[str] = Field(
        None,
        description="PART_OF_GROUP, PARENT_OF, SUBSIDIARY_OF, ASSOCIATE_OF, AFFILIATED_WITH, or JOINT_VENTURE_WITH",
    )
    ownership_percentage: Optional[float] = Field(None, description="Ownership stake, if applicable")


class FundingProfile(BaseModel):
    """HAS_FUNDING_PROFILE — qualitative characterization of a funding source."""
    characteristics: Optional[List[str]] = Field(
        None,
        description="Low Cost, High Cost, Stable, Volatile, Retail, Institutional, Diversified, Concentrated",
    )


class ExposureRelation(BaseModel):
    """HAS_ASSET_EXPOSURE_TO / HAS_LOAN_EXPOSURE_TO / EXPOSED_TO."""
    relation_type: Optional[str] = Field(
        None, description="HAS_ASSET_EXPOSURE_TO, HAS_LOAN_EXPOSURE_TO, or EXPOSED_TO"
    )
    exposure_amount: Optional[float] = Field(None, description="Exposure amount, if quantified")
    exposure_percentage: Optional[float] = Field(None, description="Exposure as a percentage of total")
    period: Optional[str] = Field(None, description="Reporting period the exposure reflects")


class MetricObservation(BaseModel):
    """A company-specific value for a reusable Metric definition (section 21 rule)."""
    company_id: Optional[str] = Field(None, description="Company the observation belongs to")
    metric_id: Optional[str] = Field(None, description="Metric definition this observes")
    value: Optional[float] = Field(None, description="Observed value")
    unit: Optional[str] = Field(None, description="Unit of the observed value")
    period: Optional[str] = Field(None, description="Reporting period")
    basis: Optional[str] = Field(None, description="Standalone, consolidated, etc.")


class CausalRelationship(BaseModel):
    """Generic causal edge: CAUSES / ENABLES / IMPROVES / REDUCES / INCREASES / CONSTRAINS /
    SUPPORTS / DEPENDS_ON / EXPOSES_TO / MITIGATES / INFLUENCES."""
    relation_type: Optional[str] = Field(
        None,
        description=(
            "CAUSES, ENABLES, IMPROVES, REDUCES, INCREASES, CONSTRAINS, SUPPORTS, "
            "DEPENDS_ON, EXPOSES_TO, MITIGATES, or INFLUENCES"
        ),
    )
    confidence: Optional[str] = Field(None, description="High, Medium, or Low")
    evidence_strength: Optional[str] = Field(
        None, description="Direct, Strong Inference, Moderate Inference, or Weak Inference"
    )
    rationale: Optional[str] = Field(None, description="Reasoning behind the causal link")


class RegulatoryRelation(BaseModel):
    """REGULATED_BY / HAS_LICENSE / SUBJECT_TO / REQUIRES."""
    relation_type: Optional[str] = Field(
        None, description="REGULATED_BY, HAS_LICENSE, SUBJECT_TO, or REQUIRES"
    )
    requirement_type: Optional[str] = Field(
        None,
        description=(
            "Capital, Liquidity, Reserve, Priority Sector, or Risk Management Requirement "
            "(when relation_type is REQUIRES or SUBJECT_TO)"
        ),
    )


class SupportedByRelation(BaseModel):
    """SUPPORTED_BY — links an entity, relationship, or claim to its Evidence."""
    evidence_id: Optional[str] = Field(None, description="Evidence record ID")


class GovernanceRole(BaseModel):
    """HOLDS_ROLE — a person's standing role at a company (persistent state, not an appointment event)."""
    title: Optional[str] = Field(None, description="e.g. CEO, CFO, Independent Director, Chairman, Promoter")
    role_category: Optional[str] = Field(None, description="Executive, Board Member, or Promoter")


class OwnershipStake(BaseModel):
    """HELD_BY — an ownership relationship carrying a stake percentage."""
    percentage: Optional[float] = Field(None, description="Percentage of equity held")
    period: Optional[str] = Field(None, description="Reporting period the stake reflects")


# ============================================================
# PLAIN DIRECTIONAL RELATIONSHIP TYPES (no extra properties)
# ============================================================

RELATIONSHIP_TYPES = [
    # Sector / structure
    "BELONGS_TO", "BELONGS_TO_SEGMENT", "OPERATES_IN_INDUSTRY", "HAS_COMPANY",
    # Products / services / customers
    "OFFERS_PRODUCT", "TARGETS", "DISTRIBUTED_THROUGH", "PROVIDES_SERVICE", "SERVES",
    # Funding / revenue
    "FUNDED_BY", "HAS_REVENUE_STREAM", "GENERATES_REVENUE_FROM",
    # Distribution / geography
    "USES_CHANNEL", "OPERATES_IN", "HAS_STRONG_PRESENCE_IN", "HAS_STRATEGIC_EXPOSURE_TO",
    # Business/operating model
    "HAS_BUSINESS_MODEL", "HAS_OPERATING_MODEL",
    # Capability / technology
    "HAS_CAPABILITY", "ENABLES", "CREATES_ADVANTAGE_IN", "USES_TECHNOLOGY",
    # Competition / advantage
    "COMPETES_WITH", "COMPETES_IN", "HAS_ADVANTAGE_IN", "DIFFERENTIATES_FROM", "HAS_ADVANTAGE", "IMPROVES",
    # SWOT
    "HAS_STRENGTH", "SUPPORTS", "HAS_WEAKNESS", "INCREASES_EXPOSURE_TO",
    "HAS_OPPORTUNITY", "DEPENDS_ON", "THREATENS", "AFFECTS",
    # Risk
    "IMPACTS", "MITIGATED_BY",
    # Dependencies
    "CREATES_RISK", "CONSTRAINS",
    # Metrics
    "HAS_METRIC", "MEASURES",
    # Drivers
    "HAS_DRIVER", "HAS_GROWTH_DRIVER", "HAS_MARGIN_DRIVER", "HAS_ROE_DRIVER",
    "HAS_RISK_DRIVER", "HAS_VALUATION_DRIVER",
    # Regulation / partnerships / ecosystem / positioning
    "PARTNERS_WITH", "PART_OF_ECOSYSTEM", "POSITIONED_AS",
    # Provenance
    "FROM",
    # Governance / ownership / ratings (extension)
    "MEMBER_OF", "RATED_BY", "HAS_RATING",
]