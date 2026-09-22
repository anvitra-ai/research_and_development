import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import asyncio


async def main():
    import os

    # Max concurrent LLM/DB operations (must be set before graphiti_core is imported)
    CONCURRENCY_LIMIT = int(os.getenv("INGEST_CONCURRENCY_LIMIT", "50"))
    os.environ["SEMAPHORE_LIMIT"] = str(CONCURRENCY_LIMIT)



    import json
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
    from graphiti_core import Graphiti
    from datetime import datetime, timezone
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

    from graphiti_core import Graphiti
    from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    import graphiti_core.helpers as graphiti_helpers

    # Apply concurrency limit (also updates module if graphiti_core was imported earlier)
    graphiti_helpers.SEMAPHORE_LIMIT = CONCURRENCY_LIMIT

    import os

    # Google API key configuration
    api_key = os.getenv("GEMINI_API_KEY")

    # Connection and model settings come from the environment (graphiti/.env),
    # matching how retrieval/src/formica_retrieval/config.py sources the same
    # values -- the Neo4j credentials used to be literals here, which meant a
    # password in source control and two places to change when the DB moved.
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin1234")

    INGEST_MODEL = os.getenv("INGEST_MODEL", "gemini-3.1-pro-preview")
    INGEST_SMALL_MODEL = os.getenv("INGEST_SMALL_MODEL", "gemini-3.6-flash")
    INGEST_EMBEDDING_MODEL = os.getenv("INGEST_EMBEDDING_MODEL", "gemini-embedding-001")

    # NOTE: GeminiClient.__init__ takes its own `max_tokens` kwarg that silently
    # overwrites whatever LLMConfig(max_tokens=...) was set to (LLMClient.__init__
    # sets self.max_tokens = config.max_tokens, then GeminiClient.__init__ immediately
    # overwrites it with its own max_tokens=None default). Without passing max_tokens
    # here explicitly, graphiti falls back to GEMINI_MODEL_MAX_TOKENS.get(model, 8192) --
    # and "gemini-3.1-pro-preview" isn't a recognized key in that table, so every
    # extraction call was silently capped at 8192 output tokens, truncating structured
    # JSON output for dense episodes (the "Unterminated string" / JSON parse errors).
    #
    # 65536 is the real output-token ceiling for Gemini 3 / 2.5 pro & flash models
    # (see graphiti_core.llm_client.gemini_client.GEMINI_MODEL_MAX_TOKENS) -- use the
    # max available rather than an arbitrary smaller number, since dense ontology
    # chunks with many entities/facts can legitimately need a lot of output tokens.
    LLM_MAX_TOKENS = int(os.getenv("INGEST_MAX_TOKENS", "65536"))

    # Initialize Graphiti with Gemini clients
    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,

        llm_client=GeminiClient(
            config=LLMConfig(
                api_key=api_key,
                model=INGEST_MODEL,              # Best reasoning model
                small_model=INGEST_SMALL_MODEL,  # Fast production model
                max_tokens=LLM_MAX_TOKENS,
            ),
            max_tokens=LLM_MAX_TOKENS,  # must also be passed here -- see note above
        ),

        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=api_key,
                embedding_model=INGEST_EMBEDDING_MODEL,
                #embedding_dim=1024                     # Verify this matches your wrapper's expected dimension
            )
        ),

        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(
                api_key=api_key,
                model=INGEST_SMALL_MODEL,
                small_model=INGEST_SMALL_MODEL,
            )
        ),

        max_coroutines=CONCURRENCY_LIMIT,
    )

    # Now you can use Graphiti with Google Gemini for all components

    # graphiti_core's own extract_edges() hardcodes max_tokens=16384 for its LLM
    # call (edge_operations.py), passed as an EXPLICIT request-level max_tokens.
    # _resolve_max_tokens() gives explicit request-level values top precedence, so
    # that 16384 overrides our client-level LLM_MAX_TOKENS entirely for edge
    # extraction -- which is exactly what truncated the edge-extraction JSON
    # ("Unterminated string...") even after raising the client default.
    #
    # There's no public knob to change that internal call, so patch the one method
    # both paths funnel through: force a floor of LLM_MAX_TOKENS under whatever
    # max_tokens any internal call explicitly requests (node extraction, edge
    # extraction, attribute hydration, etc. all go through this).
    #
    # IMPORTANT: this cell must be safe to re-run without a kernel restart. A naive
    # "save the original, then wrap it" patch breaks on a second run: the ORIGINAL
    # is stored in a global that gets overwritten by the second run's own "save"
    # step, and since the wrapper looks that global up BY NAME at call time (not at
    # definition time), the first run's wrapper ends up calling itself -> infinite
    # recursion. Fixed two ways: (1) bind the original as a default argument, which
    # Python freezes into the function object at def time rather than re-resolving
    # it from the module namespace on every call, and (2) a one-time guard so a
    # re-run of this cell is a true no-op instead of wrapping an already-patched
    # method again.
    from graphiti_core.llm_client.gemini_client import GeminiClient as _GeminiClient

    if not getattr(_GeminiClient, "_max_tokens_floor_patched", False):
        _original_resolve_max_tokens = _GeminiClient._resolve_max_tokens

        def _resolve_max_tokens_with_floor(
            self, requested_max_tokens, model, _orig=_original_resolve_max_tokens
        ):
            resolved = _orig(self, requested_max_tokens, model)
            return max(resolved, LLM_MAX_TOKENS)

        _GeminiClient._resolve_max_tokens = _resolve_max_tokens_with_floor
        _GeminiClient._max_tokens_floor_patched = True
        print(f"Patched GeminiClient._resolve_max_tokens: floor = {LLM_MAX_TOKENS} tokens")
    else:
        print(f"Already patched (floor = {LLM_MAX_TOKENS} tokens) -- skipping re-patch")


    from graphiti_core import Graphiti
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data

    # RESUME_FROM_INDEX must be decided BEFORE clear_data() runs, not after (its
    # other definition further down, near episodes_to_process, is display-only
    # now) -- clear_data() used to run unconditionally on every invocation of
    # this script, which is correct for a fresh full rebuild but is exactly
    # backwards for a resume: it would wipe the episodes already ingested by the
    # run being resumed, leaving only the tail end processed instead of the
    # union of both. Set this to 0 for a genuine full rebuild (e.g. after
    # changing the chunker or ontology in a way that invalidates everything
    # already in the graph); set it to the index a prior run stopped at to
    # continue that run instead of restarting it.
    RESUME_FROM_INDEX = int(os.getenv("INGEST_RESUME_FROM_INDEX", "0"))

    if RESUME_FROM_INDEX == 0:
        # Full clean rebuild: the previous run's chunker still injected the
        # ALL-CAPS bank-name header into episode content, which caused Gemini to
        # extract it as a separate Company entity from the body's normally-cased
        # mentions -- duplicate nodes for most banks (STATE BANK OF INDIA / State
        # Bank of India, plus bare-ticker nodes like PNB, IOB, PSB, UNIONBANK).
        # That's now fixed in the chunker above, but the existing graph already
        # has those duplicates baked in from the prior run, so a resume can't
        # clean them up -- must clear and re-ingest all 41 episodes from scratch.
        await clear_data(graphiti.driver)
    else:
        print(
            f"⏭️  RESUME_FROM_INDEX={RESUME_FROM_INDEX} -- skipping clear_data() so the "
            f"episodes already ingested by the run being resumed are kept."
        )
    #await graphiti.build_indices_and_constraints() # creates indexes once

    from pydantic import BaseModel, Field
    from datetime import datetime
    from typing import Optional, List


    # ============================================================
    # LAYER 1 — STRUCTURAL ENTITY TYPES
    # ============================================================

    class Sector(BaseModel):
        """Top-level sector classification (e.g. Financial Services)."""
        sector_name: Optional[str] = Field(None, description="Sector name")
        parent_sector: Optional[str] = Field(None, description="Parent sector, if nested")


    class Industry(BaseModel):
        """Industry within a sector (e.g. Banking)."""
        industry_name: Optional[str] = Field(None, description="Industry name")
        sector: Optional[str] = Field(None, description="Parent sector name")


    class BankingSegment(BaseModel):
        """Banking segment classification, e.g. Public Sector Bank, Private Sector Bank."""
        banking_segment_name: Optional[str] = Field(None, description="Segment name (e.g. Private Sector Banks)")
        description: Optional[str] = Field(None, max_length=800, description="Segment description")


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
        description: Optional[str] = Field(None, max_length=800, description="Free-text company description")


    class CorporateGroup(BaseModel):
        """A corporate group / conglomerate that companies may belong to."""
        corporate_group_name: Optional[str] = Field(None, description="Group name")
        description: Optional[str] = Field(None, max_length=800, description="Group description")


    class Product(BaseModel):
        """A banking product offered to customers."""
        product_name: Optional[str] = Field(None, description="Product name (e.g. Home Loan)")
        category: Optional[str] = Field(
            None,
            description="Deposit, Loan, Card, Payment, Wealth, Insurance, Forex, Treasury, or Investment Product",
        )
        description: Optional[str] = Field(None, max_length=800, description="Product description")


    class Service(BaseModel):
        """A banking service (non-product offering)."""
        service_name: Optional[str] = Field(None, description="Service name (e.g. Trade Finance)")
        description: Optional[str] = Field(None, max_length=800, description="Service description")


    class CustomerSegment(BaseModel):
        """A customer segment served by a company."""
        customer_segment_name: Optional[str] = Field(
            None,
            description=(
                "Retail, Mass Market, Affluent, HNI, UHNI, SME, Mid Market, Large Corporate, "
                "Government, Financial Institution, Agriculture, or Rural Customer"
            ),
        )
        description: Optional[str] = Field(None, max_length=800, description="Segment description")


    class Geography(BaseModel):
        """A geographic entity: country, state, region, or city."""
        geography_name: Optional[str] = Field(None, description="Geography name")
        level: Optional[str] = Field(None, description="Country, State, Region, or City")
        area_type: Optional[str] = Field(None, description="Urban, Semi Urban, Rural, or International Market")
        parent_geography: Optional[str] = Field(None, description="Parent geography name, if nested")


    class DistributionChannel(BaseModel):
        """A channel through which products/services reach customers."""
        distribution_channel_name: Optional[str] = Field(
            None,
            description=(
                "Branch, ATM, Mobile Application, Web Platform, API, Relationship Manager, "
                "Direct Sales, Agent Partner, Merchant Network, or Corporate Partnership"
            ),
        )


    class Regulator(BaseModel):
        """A regulatory body."""
        regulator_name: Optional[str] = Field(None, description="Regulator name (e.g. RBI, SEBI)")
        jurisdiction: Optional[str] = Field(None, description="Jurisdiction covered")


    class License(BaseModel):
        """A license or authorization held by a company."""
        license_name: Optional[str] = Field(None, description="License name/type")
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
        description: Optional[str] = Field(None, max_length=800, description="Description of this business model dimension")


    class OperatingModel(BaseModel):
        """A company's operating model archetype."""
        operating_model_name: Optional[str] = Field(
            None,
            description="Branch Led, Digital First, Hybrid, Relationship Led, Partnership Led, or Platform Led",
        )


    class RevenueStream(BaseModel):
        """A source of revenue for a company."""
        revenue_stream_name: Optional[str] = Field(
            None,
            description=(
                "Interest Income, Fee Income, Commission Income, Treasury Income, Forex Income, "
                "Payment Income, Wealth Income, Insurance Distribution Income, or Other Operating Income"
            ),
        )


    class FundingSource(BaseModel):
        """A source of funding/liabilities for a company."""
        funding_source_name: Optional[str] = Field(
            None,
            description=(
                "CASA, Savings Deposits, Current Deposits, Term Deposits, Institutional Deposits, "
                "Borrowings, Bonds, or Interbank Funding"
            ),
        )


    class AssetClass(BaseModel):
        """A category of asset held by a company."""
        asset_class_name: Optional[str] = Field(
            None, description="Loan Portfolio, Investments, Cash, Interbank Assets, or Other Assets"
        )


    class LiabilityClass(BaseModel):
        """A category of liability held by a company."""
        liability_class_name: Optional[str] = Field(None, description="Liability class name")


    class LoanSegment(BaseModel):
        """A loan portfolio segment."""
        loan_segment_name: Optional[str] = Field(
            None,
            description=(
                "Retail Lending (Home/Vehicle/Personal/Credit Card/Education Loan) or Corporate "
                "Lending (Large/Mid Corporate, SME, Agriculture, Infrastructure Lending)"
            ),
        )
        segment_group: Optional[str] = Field(None, description="Retail Lending or Corporate Lending")


    class DepositSegment(BaseModel):
        """A deposit portfolio segment."""
        deposit_segment_name: Optional[str] = Field(None, description="Deposit segment name")


    class InvestmentClass(BaseModel):
        """A class of investment holding."""
        investment_class_name: Optional[str] = Field(None, description="Investment class name")


    class Capability(BaseModel):
        """An operational capability a company possesses."""
        capability_name: Optional[str] = Field(
            None,
            description=(
                "Underwriting, Risk Management, Fraud Detection, Customer Acquisition, Customer "
                "Retention, Cross Selling, Collections, Data Analytics, Relationship Management, "
                "Treasury Management, Digital Onboarding, or Payment Processing"
            ),
        )


    class Technology(BaseModel):
        """A technology category used by a company."""
        technology_name: Optional[str] = Field(
            None,
            description=(
                "Core Banking, Cloud, Artificial Intelligence, Data Platform, Cybersecurity, "
                "Mobile Banking, API, Payment Infrastructure, or Automation"
            ),
        )


    class CompetitiveArena(BaseModel):
        """A market/arena in which companies compete."""
        competitive_arena_name: Optional[str] = Field(
            None,
            description=(
                "Deposits, Retail Lending, Corporate Lending, SME Lending, Credit Cards, Payments, "
                "Wealth Management, Digital Banking, or Transaction Banking"
            ),
        )


    class Partnership(BaseModel):
        """A persistent partnership entity (not a one-off partnership event)."""
        partnership_name: Optional[str] = Field(None, description="Partnership name/description")
        category: Optional[str] = Field(
            None, description="Technology, Distribution, Lending, Payments, Insurance, Wealth, or Data"
        )


    class Ecosystem(BaseModel):
        """An ecosystem a company participates in."""
        ecosystem_name: Optional[str] = Field(
            None,
            description=(
                "Banking, Insurance, Asset Management, Broking, Payments, Lending, Wealth, "
                "Consumer, or Platform"
            ),
        )


    class Metric(BaseModel):
        """A reusable metric definition (company-specific values live in MetricObservation)."""
        metric_name: Optional[str] = Field(None, description="Metric name, e.g. ROA, ROE, NIM, GNPA, CET1")
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
        person_name: Optional[str] = Field(None, description="Full name")
        role_category: Optional[str] = Field(None, description="Executive, Board Member, or Promoter")


    class PromoterGroup(BaseModel):
        """A promoter entity (individual or family group) holding a controlling stake."""
        promoter_group_name: Optional[str] = Field(None, description="Promoter group name")
        description: Optional[str] = Field(None, max_length=800, description="Description of the promoter group")


    class Shareholder(BaseModel):
        """A category of institutional shareholder holding equity in a company."""
        shareholder_name: Optional[str] = Field(None, description="Shareholder name, if a specific institution")
        holder_type: Optional[str] = Field(
            None, description="Promoter, FII, DII, Mutual Fund, Insurance, or Retail"
        )


    class CreditRatingAgency(BaseModel):
        """A credit rating agency, e.g. CRISIL, ICRA, Moody's, S&P."""
        credit_rating_agency_name: Optional[str] = Field(None, description="Rating agency name")


    class CreditRating(BaseModel):
        """A specific credit rating assigned to a company or capital instrument."""
        scale_value: Optional[str] = Field(None, description="Rating symbol, e.g. AAA, AA+, Baa1")
        outlook: Optional[str] = Field(None, description="Stable, Positive, Negative, or Under Review")
        rated_entity_type: Optional[str] = Field(None, description="Issuer rating or Instrument rating")
        period: Optional[str] = Field(None, description="Date/period the rating reflects")


    class MacroeconomicFactor(BaseModel):
        """An external macroeconomic or policy factor relevant to bank performance."""
        macroeconomic_factor_name: Optional[str] = Field(
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
        statement: Optional[str] = Field(None, max_length=800, description="Description of the strength")


    class Weakness(BaseModel):
        """An analytical weakness attributed to a company."""
        category: Optional[str] = Field(
            None,
            description=(
                "Financial, Funding, Distribution, Technology, Customer Concentration, "
                "Geographic Concentration, Product Concentration, Scale, or Capability"
            ),
        )
        statement: Optional[str] = Field(None, max_length=800, description="Description of the weakness")


    class Opportunity(BaseModel):
        """An analytical opportunity attributed to a company."""
        category: Optional[str] = Field(
            None,
            description=(
                "Market/Customer/Product/Geographic/Digital/Margin Expansion, Cross Sell, "
                "Partnership, Ecosystem, or Regulatory Opportunity"
            ),
        )
        statement: Optional[str] = Field(None, max_length=800, description="Description of the opportunity")


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
        statement: Optional[str] = Field(None, max_length=800, description="Description of the threat")


    class Risk(BaseModel):
        """A risk category a company is exposed to."""
        category: Optional[str] = Field(
            None,
            description=(
                "Credit, Market, Liquidity, Interest Rate, Operational, Cyber, Regulatory, "
                "Concentration, Reputation, Technology, or Funding Risk"
            ),
        )
        description: Optional[str] = Field(None, max_length=800, description="Risk description")


    class Dependency(BaseModel):
        """A structural dependency a company has."""
        category: Optional[str] = Field(
            None,
            description=(
                "Technology, Funding, Partner, Customer, Vendor, Geographic, Distribution, "
                "or Regulatory Dependency"
            ),
        )
        description: Optional[str] = Field(None, max_length=800, description="Dependency description")


    class Driver(BaseModel):
        """A causal driver of growth, revenue, margin, profitability, ROE, risk, competitiveness, or valuation."""
        driver_type: Optional[str] = Field(
            None,
            description=(
                "Growth, Revenue, Margin, Profitability, ROE, Risk, Competitive, Structural, "
                "or Valuation Driver"
            ),
        )
        statement: Optional[str] = Field(None, max_length=800, description="Description of the driver")


    class CompetitiveAdvantage(BaseModel):
        """A source of competitive advantage."""
        source: Optional[str] = Field(
            None,
            description=(
                "Brand, Distribution, Scale, Cost of Funds, Customer Base, Data, Technology, "
                "Ecosystem, Regulatory License, Network Effects, or Switching Costs"
            ),
        )
        description: Optional[str] = Field(None, max_length=800, description="Description of the advantage")


    class StrategicPosition(BaseModel):
        """A company's strategic positioning."""
        strategic_position_name: Optional[str] = Field(
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
        statement: Optional[str] = Field(None, max_length=800, description="The analytical statement being made")
        subject_entity_id: Optional[str] = Field(None, description="Entity the claim is about")
        supporting_entity_ids: Optional[List[str]] = Field(
            None, description="Entities cited in support of the claim"
        )
        confidence: Optional[str] = Field(None, description="High, Medium, or Low")
        evidence_strength: Optional[str] = Field(
            None, description="Direct, Strong Inference, Moderate Inference, or Weak Inference"
        )
        rationale: Optional[str] = Field(None, max_length=800, description="Reasoning behind the claim")


    class Condition(BaseModel):
        """A precondition referenced within analyst reasoning (e.g. a causal chain)."""
        description: Optional[str] = Field(None, max_length=800, description="Condition description")


    class Outcome(BaseModel):
        """An analytical outcome referenced within analyst reasoning."""
        description: Optional[str] = Field(None, max_length=800, description="Outcome description")


    class Evidence(BaseModel):
        """Provenance record supporting an entity, relationship, metric, or claim."""
        evidence_id: Optional[str] = Field(None, description="Stable, unique evidence identifier")
        entity_or_claim_id: Optional[str] = Field(None, description="ID of the entity/claim this supports")
        evidence_type: Optional[str] = Field(None, description="Type of evidence, e.g. filing, disclosure, report")
        source_name: Optional[str] = Field(None, description="Name of the source")
        source_reference: Optional[str] = Field(None, description="Reference/citation for the source")
        description: Optional[str] = Field(None, max_length=800, description="Description of what the evidence shows")
        confidence: Optional[str] = Field(None, description="Confidence in the evidence itself")


    class Source(BaseModel):
        """A named source referenced by Evidence records."""
        source_name: Optional[str] = Field(None, description="Source name")
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
        # Source prose routinely states a level and its growth in one breath --
        # "total deposits were Rs 60.06 trillion, up 9.73% year-on-year". With no
        # field for the second number the extractor kept the level and silently
        # dropped the growth rate, which is precisely the figure the benchmark's
        # "deposit growth / loan book growth" questions ask for. Measured against
        # the source file, only 62% of growth percentages reached the graph.
        change_percent: Optional[float] = Field(
            None,
            description=(
                "Percentage change in this metric over the comparison period, when the text "
                "states one (e.g. 9.73 for 'up 9.73% year-on-year'). Record this IN ADDITION "
                "to value -- never choose between the level and its growth rate."
            ),
        )
        change_period: Optional[str] = Field(
            None,
            description="Comparison basis for change_percent, e.g. 'year-on-year', 'quarter-on-quarter'",
        )


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
        rationale: Optional[str] = Field(None, max_length=800, description="Reasoning behind the causal link")


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

    # from kg.ontology.banking.banking_ontology_models import (
    #     # Layer 1 — structural entities
    #     Sector, Industry, BankingSegment, Company, CorporateGroup, Product, Service,
    #     CustomerSegment, Geography, DistributionChannel, Regulator, License,
    #     # Layer 2 — business and economic entities
    #     BusinessModel, OperatingModel, RevenueStream, FundingSource, AssetClass,
    #     LiabilityClass, LoanSegment, DepositSegment, InvestmentClass, Capability,
    #     Technology, CompetitiveArena, Partnership, Ecosystem, Metric,
    #     # Layer 3 — analyst entities
    #     Strength, Weakness, Opportunity, Threat, Risk, Dependency, Driver,
    #     CompetitiveAdvantage, StrategicPosition, Claim, Condition, Outcome,
    #     Evidence, Source,
    #     # Governance / ownership / ratings extension
    #     Person, PromoterGroup, Shareholder, CreditRatingAgency, CreditRating,
    #     MacroeconomicFactor,
    #     # Edge types that carry their own properties
    #     CorporateStructureRelation, FundingProfile, ExposureRelation,
    #     MetricObservation, CausalRelationship, RegulatoryRelation,
    #     SupportedByRelation, Relationship, GovernanceRole, OwnershipStake,
    # )

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
        # Direct Entity -> Source shortcut, alongside the 2-hop Entity ->
        # Evidence -> Source path above. The source text cites its sources as
        # a simple inline "(source: Investing.com's Q1 FY27 slides, August
        # 2026)" tag on a fact, not as a separately-claimed, structured
        # Evidence record -- 104 such citations appear across the document,
        # but under custom_extraction_instructions' "omit rather than invent a
        # new fact type" rule, requiring an extra intermediate Evidence node
        # for every single one made the 2-hop path too heavy for the LLM to
        # consistently take: only ~6 SupportedByRelation edges of any kind
        # made it into the graph on the first ingestion (measured directly).
        # This direct path gives the LLM a lightweight way to record "this
        # fact/entity is attributed to source X" that matches how the
        # citation actually reads, without discarding the richer Evidence-node
        # path above for cases where the LLM does have enough structure to
        # populate a fuller record (confidence, evidence_type, etc.).
        ("Entity", "Source"): ["SupportedByRelation"],

        # Generic causal connectors — apply broadly across the graph,
        # not tied to a specific entity pair (RTF section 23)
        ("Entity", "Entity"): [
            "CausalRelationship", "ENABLES", "SUPPORTS", "IMPROVES",
            "CONSTRAINS", "DEPENDS_ON", "INFLUENCES",
        ],
    }




    import httpx
    from google.genai.errors import ServerError


    # Transient failures shouldn't kill a 61-episode ingestion run, so retry these
    # with backoff before giving up:
    #   - httpx.TransportError / ConnectionError / OSError: dropped Wi-Fi/VPN,
    #     momentary DNS blip, connection reset mid-request -- can surface anywhere
    #     inside graphiti.add_episode() (LLM calls, embedding calls, the Neo4j
    #     driver itself).
    #   - google.genai.errors.ServerError: Gemini's own 5xx responses (500, 502,
    #     503 "The service is currently unavailable", 504) -- transient overload
    #     on Google's side, not something wrong with our request.
    # google.genai.errors.ClientError (4xx: bad request, invalid API key, quota
    # exceeded, ...) is deliberately NOT included -- those won't succeed on retry.
    # A non-transient error (EntityTypeValidationError, a malformed episode, ...)
    # is NOT one of these types and propagates immediately on the first attempt,
    # same as before.
    _TRANSIENT_ERRORS = (httpx.TransportError, ConnectionError, OSError, ServerError)


    async def _add_episode_with_retry(
        graphiti, max_retries=6, base_delay_seconds=15, **add_episode_kwargs
    ):
        """Call graphiti.add_episode(**add_episode_kwargs), retrying transient
        network/server failures with exponential backoff (10s, 20s, 40s, 80s by
        default) before re-raising. Non-transient errors propagate immediately
        without retrying.
        """
        for attempt in range(max_retries + 1):
            try:
                return await graphiti.add_episode(**add_episode_kwargs)
            except Exception as e:
                # graphiti_core itself re-raises internal LLM failures (after its
                # own retries are exhausted) as a bare Exception (edge_operations.py
                # / gemini_client.py do "raise Exception from e"), discarding the
                # real exception type -- so a genuinely transient Gemini 503 is
                # indistinguishable here from _TRANSIENT_ERRORS by type alone.
                # Retry broadly at this level too (still bounded by max_retries)
                # rather than aborting the whole batch on what is very likely the
                # same kind of transient failure the narrower tuple was meant to catch.
                if attempt == max_retries:
                    print(f"❌ Giving up after {max_retries + 1} attempts: {e!r}")
                    raise
                delay = base_delay_seconds * (2 ** attempt)
                print(
                    f"⚠️  Transient error (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{e!r} -- retrying in {delay}s"
                )
                await asyncio.sleep(delay)


    async def add_episodes_to_graph(
        graphiti,
        episodes,
        group_id,
        prefix="Episode",
        excluded_entity_types=None,
        custom_extraction_instructions=None,
    ):
        """Add a list of episodes to the graph using Graphiti.

        excluded_entity_types : list[str] | None
            Passed through to graphiti.add_episode. Pass ["Entity"] to enforce strict
            ontology typing — any extracted entity that the LLM can't classify into one
            of `entity_types` is dropped instead of being kept as a generic "Entity" node.
        custom_extraction_instructions : str | None
            Passed through to graphiti.add_episode to steer the extraction prompt (e.g.
            to forbid inventing entity/edge types outside the declared ontology).

        Each episode's add_episode() call is retried with backoff on transient network
        failures (see _add_episode_with_retry) so a momentary connection drop doesn't
        abort the whole batch.

        Returns the list of AddEpisodeResults, one per episode, so callers can inspect
        or validate the nodes/edges actually written (see validate_ontology_conformance).
        """
        print(f"📝 Adding {len(episodes)} episodes to graph...")

        results = []
        for i, episode in enumerate(episodes):
            name = episode.get('name', f"{prefix} {i+1}")
            content = episode['content']

            # Convert non-string content to JSON
            if not isinstance(content, str):
                content = json.dumps(content)

            # Graphiti method for addin data
            result = await _add_episode_with_retry(
                graphiti,
                name=name,
                episode_body=content,
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
                entity_types=entity_types,
                excluded_entity_types=excluded_entity_types,
                edge_types=edge_types,
                edge_type_map=edge_type_map,
                custom_extraction_instructions=custom_extraction_instructions,
            )
            results.append(result)

        print(f"✅ Successfully added {len(episodes)} episodes!")
        return results


    def get_article_from_url(url):
        """Scrape article content from a URL."""
        print(f"📰 Fetching article from: {url}")

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract date
        date_meta = soup.find("meta", {"name": "DC.date.issued"})
        article_date = date_meta["content"] if date_meta and date_meta.get("content") else "Date not found"

        # Extract article text
        paragraphs = soup.find_all("p")
        filtered = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
        article_text = "\n\n".join(filtered).encode("utf-8", "ignore").decode("utf-8")
        article_text = article_text.replace("í", "i")

        print("✅ Article extracted successfully")
        return article_date, article_text


    async def search_and_display(graphiti, query, num_results=3,search_filter=None):
        """Search the graph and display results in a clean format."""
        print(f"🔍 Searching for: '{query}'")
        print("-" * 50)

        results = await graphiti.search(query, num_results=num_results,search_filter=search_filter)

        for i, r in enumerate(results, 1):
            print(f"{i}. {r.fact}")
            print(f"   Label: {r.name}")
            print(f"   📅 Valid from: {r.valid_at}")
            if r.invalid_at:
                print(f"   ❌ Invalid at: {r.invalid_at}")
            print()

        return results

    async def add_data (episodes):
    # Add episodes to the graph
      for episode in episodes:
          if episode[ 'type'] == EpisodeType.json:
              episode['content'] = json. dumps(episode['content' ])
          await graphiti.add_episode(name=episode['name'],
                                     episode_body=episode['content' ],
                                     source=episode['type'],
                                     source_description=episode['description'],
                                     reference_time=datetime.now(timezone.utc),
                                 
                                      )
          print(f'Added episode: {episode["name"]}')

    import re
    from pathlib import Path

    DATA_FILE = Path("data/ontology_data/banking_ontology_data.txt")
    MAX_CHARS = 6000   # keep each episode small enough for reliable LLM extraction


    def load_text_as_episodes(path, max_chars=MAX_CHARS):
        """Split the banking ontology text into episodes, one per bank entry.

        The source file uses plain-text delimiters, not markdown: "===...==="
        banner lines mark PART sections (Public/Private/Small Finance Banks) and
        a lone "---" line separates each bank's entry within a part. (An earlier
        version of this function assumed markdown "##"/"###" headers, which never
        match here -- the whole file collapsed into a single block and got sliced
        into arbitrary ~6,000-character chunks that routinely split mid-sentence
        across two unrelated banks, with no company name carried into the chunk
        context. That's why extraction of qualitative facts -- competitive
        advantage, strength, weakness, threat -- was so inconsistent: whichever
        bank's paragraph happened to land fully inside one arbitrary chunk got
        captured, and whichever got cut across a chunk boundary or lost its
        antecedent ("the bank") to the previous chunk did not.)

        Splitting on the real delimiters instead means every episode is one
        bank's complete, self-contained entry, correctly titled -- confirmed to
        keep every bank comfortably under max_chars with no sub-splitting needed
        for this dataset, though the paragraph-aligned sub-chunking below still
        guards against a future bank entry that grows past the limit.
        """
        raw = Path(path).read_text(encoding="utf-8").strip()

        part_re = re.compile(r"^=+\s*\n(.+?)\n=+\s*$", re.MULTILINE)
        part_matches = list(part_re.finditer(raw))

        sections = []
        if part_matches:
            preamble = raw[: part_matches[0].start()].strip()
            if preamble:
                sections.append(("Preamble", preamble))
            for i, m in enumerate(part_matches):
                end = part_matches[i + 1].start() if i + 1 < len(part_matches) else len(raw)
                sections.append((m.group(1).strip(), raw[m.end() : end].strip()))
        else:
            sections = [("Preamble", raw)]

        episodes_array = []
        for part, body in sections:
            if not body:
                continue
            bank_blocks = [b.strip() for b in re.split(r"\n-{3,}\n", body) if b.strip()]

            for block in bank_blocks:
                lines = block.splitlines()
                title = lines[0].strip()
                # Drop the ALL-CAPS header line ("STATE BANK OF INDIA (ticker
                # SBIN, NSE and BSE)") from what's actually sent to extraction --
                # keep it only in the episode `name` for tracking. The body's very
                # next sentence always restates the company name correctly cased
                # ("State Bank of India is India's largest bank..."), so the
                # header added nothing extraction needed. Left in, it did active
                # harm: Gemini extracted the ALL-CAPS header as one entity and the
                # body's normally-cased mentions as a second, separate one --
                # producing duplicate Company nodes for most banks (confirmed
                # after the first full re-ingestion: STATE BANK OF INDIA / State
                # Bank of India, IDBI BANK LIMITED / IDBI Bank, etc., plus bare
                # ticker-only nodes like PNB and IOB splitting off as their own
                # "companies") and measurably regressing retrieval quality.
                #
                # BUT: the ticker symbol and exchange listing are stated ONLY in
                # that header's parenthetical ("(ticker SBIN, NSE and BSE)") --
                # confirmed by grepping the source file, e.g. "SBIN" appears
                # nowhere else in the whole document. Dropping the header
                # wholesale silently took this with it: after the header-fix
                # re-ingestion, only 3 of 68 companies had a populated
                # Company.ticker property, vs. virtually every company having it
                # in the source text (measured via the grounded benchmark: 38/38
                # ticker/exchange queries failed, a 100% rate found for no other
                # query category -- the tell that this is a systematic
                # extraction miss, not ordinary per-company source-text
                # sparsity). Fix: keep just the parenthetical (not the ALL-CAPS
                # name, so the duplicate-entity bug doesn't come back) and
                # prepend it on its own line -- the body's next sentence still
                # supplies the properly-cased company name for context, same as
                # for every other fact in this episode.
                ticker_match = re.search(r"\(ticker[^)]*\)", title, re.I)
                content_block = "\n".join(lines[1:]).strip()
                if ticker_match:
                    content_block = f"{ticker_match.group(0)}\n\n{content_block}"

                # Break oversized entries into paragraph-aligned chunks (not
                # expected to trigger for this dataset -- see docstring -- but
                # kept as a safety net).
                chunks, current = [], ""
                for para in content_block.split("\n\n"):
                    if current and len(current) + len(para) + 2 > max_chars:
                        chunks.append(current)
                        current = para
                    else:
                        current = f"{current}\n\n{para}" if current else para
                if current:
                    chunks.append(current)

                for i, chunk in enumerate(chunks):
                    suffix = f" (part {i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                    episodes_array.append({
                        "name": f"[{part}] {title}{suffix}",
                        # No ALL-CAPS title re-injected into continuation chunks
                        # either, for the same reason it was dropped above -- not
                        # expected to trigger for this dataset (see docstring),
                        # kept plain as a safety net rather than reintroducing
                        # the duplicate-entity risk if a future edit ever does.
                        "content": chunk,
                        "type": EpisodeType.text,
                        "description": "Indian banking sector knowledge graph ontology instance",
                    })

        return episodes_array


    episodes_array = load_text_as_episodes(DATA_FILE)

    print(f"📄 {DATA_FILE.name}: {len(episodes_array)} episodes")
    print(f"   total chars: {sum(len(e['content']) for e in episodes_array):,}")
    print(f"   largest episode: {max(len(e['content']) for e in episodes_array):,} chars\n")
    for e in episodes_array[:5]:
        print(f" - {e['name']} ({len(e['content'])} chars)")

    # ============================================================
    # ENFORCE the ontology at extraction time:
    #  - excluded_entity_types=["Entity"] drops any node the LLM can't
    #    classify into one of our 47 declared entity types (no generic
    #    catch-all nodes survive into the graph).
    #  - custom_extraction_instructions tells the LLM not to invent
    #    entity/edge labels outside the declared ontology.
    # Non-conforming EDGES (e.g. a relation name outside edge_types, or an
    # edge between a type pair not covered by edge_type_map) aren't rejected
    # by graphiti itself, so they're checked and reported in the next cell.
    # ============================================================

    STRICT_ONTOLOGY_INSTRUCTIONS = (
        "Strict ontology mode: you MUST classify every extracted entity into one of the "
        "provided entity types, and every relationship into one of the provided edge/fact "
        "types. Do not invent new entity types or new relationship/fact names that are not "
        "in the provided lists. If a mention does not clearly fit one of the provided entity "
        "types, extract it as the generic 'Entity' type rather than inventing a new type name. "
        "If a relationship does not clearly fit one of the provided edge/fact types for that "
        "pair of entities, omit the relationship entirely rather than inventing a new fact type. "
        "Every inline citation of the form '(source: X)' attached to a fact MUST be captured: "
        "create a Source entity named X (if one doesn't already exist) and a SupportedByRelation "
        "edge directly from the fact's subject entity to that Source entity. Do not skip this "
        "just because the citation is a short parenthetical rather than its own sentence. "
        "\n\n"
        "QUANTITATIVE COMPLETENESS -- this is mandatory, not best-effort. Every number in the "
        "text that describes a company MUST become a MetricObservation edge from that company. "
        "Extract exhaustively, including all of: deposits, advances/loan book, net interest "
        "income, net interest margin, CASA ratio, gross and net NPA, provision coverage, capital "
        "adequacy/CRAR, CET1, return on assets, return on equity, cost-to-income, slippage, "
        "branch and ATM counts, employee counts, and shareholding percentages. A metrics "
        "paragraph that lists eight figures must yield eight MetricObservation edges -- do not "
        "summarise, do not keep only the ones that seem most important, and do not stop early "
        "because the paragraph is long. "
        "\n"
        "When a sentence gives BOTH a level and its growth rate -- 'total deposits were "
        "Rs 60.06 trillion, up 9.73% year-on-year' -- record ONE observation carrying both: "
        "value=60.06 with unit='trillion INR', change_percent=9.73, change_period='year-on-year'. "
        "Dropping the growth rate because the level was already captured is the single most "
        "common extraction error on this corpus and must not happen. "
        "\n"
        "Qualitative facts (headquarters, business model, leadership) are necessary but NOT "
        "sufficient: an episode that yields only descriptive edges and no metrics has failed, "
        "because every bank section in this corpus contains quantitative disclosures."
    )

    # ============================================================
    # RESUME SUPPORT: RESUME_FROM_INDEX itself is now set once, early, right
    # before the clear_data() gate near the top of this function -- it has to
    # be decided before that call, not here, or clear_data() would run
    # unconditionally and wipe whatever a resume is trying to preserve. This is
    # just where it's applied to the episode list.
    #
    #   RESUME_FROM_INDEX = 0    -> process all episodes_array (default, no skip)
    #   RESUME_FROM_INDEX = 12   -> skip episodes_array[:12], start at episode 13/41
    # ============================================================
    episodes_to_process = episodes_array[RESUME_FROM_INDEX:]

    if RESUME_FROM_INDEX:
        print(
            f"⏭️  Resuming from episode {RESUME_FROM_INDEX + 1}/{len(episodes_array)} "
            f"-- skipping the first {RESUME_FROM_INDEX} already-processed episode(s), "
            f"{len(episodes_to_process)} remaining"
        )
    else:
        print(f"Processing all {len(episodes_array)} episodes (no skip)")

    group_id = "banking-ontology"
    results = await add_episodes_to_graph(
        graphiti,
        episodes_to_process,
        group_id,
        prefix="Banking Ontology",
        excluded_entity_types=["Entity"],
        custom_extraction_instructions=STRICT_ONTOLOGY_INSTRUCTIONS,
    )


if __name__ == "__main__":
    asyncio.run(main())
