"""Resolve DeBERTa template labels to KG Cypher query specs.

LEGACY: superseded by formica_retrieval.template_resolver (src/formica_retrieval/).
Not imported by the current batch pipeline or API; kept for reference only. Its
`from paths import ...` import is stale and won't resolve as-is.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from paths import DATA_DIR

DEFAULT_TEMPLATE_PATH = DATA_DIR / "kg_query_templates.json"

GULF_GEO_IDS = {"G_GCC", "G_QATAR", "G_OMAN", "G_RUSSIA"}
CHOKEPOINT_GEO_ID = "G_HORMUZ"

# Primary Cypher used when only one entity is resolved (replaces multi-entity template).
SINGLE_ENTITY_CYPHER: dict[str, dict[str, Any]] = {
    "T_StockImpact": {
        "COMPANY": (
            "MATCH (company:Node {id: $entity_id})-[r:HURT_BY|BENEFITS_FROM]->(risk:Node)\n"
            "RETURN company.label AS source_name, risk.label AS target_name, "
            "[company.label, risk.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "MACRO_VAR": (
            "MATCH (company:Node {type: 'COMPANY'})-[r:HURT_BY|BENEFITS_FROM]->(macro:Node {id: $entity_id})\n"
            "RETURN company.label AS source_name, macro.label AS target_name, "
            "[company.label, macro.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "EVENT": (
            "MATCH (company:Node {type: 'COMPANY'})-[r:HURT_BY|BENEFITS_FROM]->(event:Node {id: $entity_id})\n"
            "RETURN company.label AS source_name, event.label AS target_name, "
            "[company.label, event.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "GEOGRAPHY": (
            "MATCH (company:Node {type: 'COMPANY'})-[r:HURT_BY|BENEFITS_FROM|NEEDS|SOURCED_FROM]->(geo:Node {id: $entity_id})\n"
            "RETURN company.label AS source_name, geo.label AS target_name, "
            "[company.label, geo.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "default": (
            "MATCH (entity:Node {id: $entity_id})-[r:HURT_BY|BENEFITS_FROM|CAUSES]->(linked:Node)\n"
            "RETURN entity.label AS source_name, linked.label AS target_name, "
            "[entity.label, linked.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
    },
    "T_CausalChain": {
        "default": (
            "MATCH (source:Node {id: $entity_id})"
            "-[r:CAUSES|TRANSITS|HURT_BY|BENEFITS_FROM|NEEDS|PRODUCES|CONTAINS|PRICE_LINKED_TO|SOURCED_FROM]->"
            "(target:Node)\n"
            "RETURN source.label AS source_name, target.label AS target_name, "
            "[source.label, target.label] AS path_names, [type(r)] AS rel_types, 1 AS hops\n"
            "UNION\n"
            "MATCH (source:Node)"
            "-[r:CAUSES|TRANSITS|HURT_BY|BENEFITS_FROM|NEEDS|PRODUCES|CONTAINS|PRICE_LINKED_TO|SOURCED_FROM]->"
            "(target:Node {id: $entity_id})\n"
            "RETURN source.label AS source_name, target.label AS target_name, "
            "[source.label, target.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
    },
    "T_CompareExposure": {
        "COMPANY": (
            "MATCH (company:Node {id: $entity_id})-[r:HURT_BY|BENEFITS_FROM]->(risk:Node)\n"
            "RETURN company.label AS company_a, risk.label AS shock_name, "
            "type(r) AS relationship, risk.type AS risk_type"
        ),
        "MACRO_VAR": (
            "MATCH (company:Node {type: 'COMPANY'})-[r:HURT_BY|BENEFITS_FROM]->(shock:Node {id: $entity_id})\n"
            "RETURN company.label AS company_a, shock.label AS shock_name, "
            "type(r) AS relationship, shock.type AS risk_type"
        ),
        "EVENT": (
            "MATCH (company:Node {type: 'COMPANY'})-[r:HURT_BY|BENEFITS_FROM]->(shock:Node {id: $entity_id})\n"
            "RETURN company.label AS company_a, shock.label AS shock_name, "
            "type(r) AS relationship, shock.type AS risk_type"
        ),
        "GEOGRAPHY": (
            "MATCH (company:Node {type: 'COMPANY'})-[r:HURT_BY|BENEFITS_FROM|NEEDS|SOURCED_FROM]->"
            "(geo:Node {id: $entity_id})\n"
            "RETURN company.label AS company_a, geo.label AS shock_name, "
            "type(r) AS relationship, geo.type AS risk_type"
        ),
        "default": (
            "MATCH (entity:Node {id: $entity_id})-[r:HURT_BY|BENEFITS_FROM|CAUSES]->(linked:Node)\n"
            "RETURN entity.label AS company_a, linked.label AS shock_name, "
            "type(r) AS relationship, linked.type AS risk_type"
        ),
    },
    "T_MacroTransmission": {
        "COMPANY": (
            "MATCH (company:Node {id: $entity_id})-[r:HURT_BY|BENEFITS_FROM|PRICE_LINKED_TO]->(macro:Node)\n"
            "RETURN macro.label AS macro_name, company.label AS company_name, "
            "[macro.label, company.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "MACRO_VAR": (
            "MATCH (macro:Node {id: $entity_id})-[*1..6]-(company:Node {type: 'COMPANY'})\n"
            "RETURN macro.label AS macro_name, company.label AS company_name, "
            "[macro.label, company.label] AS path_names, ['PATH'] AS rel_types, 1 AS hops\n"
            "LIMIT 20"
        ),
        "EVENT": (
            "MATCH (event:Node {id: $entity_id})-[r:CAUSES|HURT_BY]->(impacted:Node)\n"
            "RETURN event.label AS macro_name, impacted.label AS company_name, "
            "[event.label, impacted.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "GEOGRAPHY": (
            "MATCH (geo:Node {id: $entity_id})<-[r:HURT_BY|BENEFITS_FROM|NEEDS|SOURCED_FROM]-(company:Node {type: 'COMPANY'})\n"
            "RETURN geo.label AS macro_name, company.label AS company_name, "
            "[geo.label, company.label] AS path_names, [type(r)] AS rel_types, 1 AS hops"
        ),
        "default": (
            "MATCH (entity:Node {id: $entity_id})-[*1..6]-(linked:Node {type: 'COMPANY'})\n"
            "RETURN entity.label AS macro_name, linked.label AS company_name, "
            "[entity.label, linked.label] AS path_names, ['PATH'] AS rel_types, 1 AS hops\n"
            "LIMIT 20"
        ),
    },
    "T_TransitRisk": {
        "RAW_MATERIAL": (
            "MATCH (commodity:Node {id: $entity_id})-[r:TRANSITS|SOURCED_FROM]->(geo:Node)\n"
            "RETURN commodity.label AS commodity_name, commodity.type AS commodity_type, "
            "type(r) AS relationship, geo.label AS chokepoint"
        ),
        "PRODUCT": (
            "MATCH (commodity:Node {id: $entity_id})-[r:TRANSITS|SOURCED_FROM|NEEDS]->(geo:Node)\n"
            "RETURN commodity.label AS commodity_name, commodity.type AS commodity_type, "
            "type(r) AS relationship, geo.label AS chokepoint"
        ),
        "GEOGRAPHY": (
            "MATCH (commodity:Node)-[r:TRANSITS|SOURCED_FROM]->(geo:Node {id: $entity_id})\n"
            "RETURN commodity.label AS commodity_name, commodity.type AS commodity_type, "
            "type(r) AS relationship, geo.label AS chokepoint, r.channel AS channel"
        ),
        "default": (
            "MATCH (commodity:Node)-[r:TRANSITS]->(geo:Node {id: $entity_id})\n"
            "RETURN commodity.label AS commodity_name, commodity.type AS commodity_type, "
            "type(r) AS relationship, geo.label AS chokepoint"
        ),
    },
    "T_SectorExposure": {
        "COMPANY": (
            "MATCH (sector:Node)-[:CONTAINS]->(company:Node {id: $entity_id})\n"
            "RETURN sector.label AS sector_name, company.label AS name, company.id AS id"
        ),
        "default": (
            "MATCH (sector:Node {id: $entity_id})-[:CONTAINS]->(company:Node {type: 'COMPANY'})\n"
            "RETURN sector.label AS sector_name, company.label AS name, company.id AS id\n"
            "ORDER BY company.label"
        ),
    },
    "T_InvestmentFlow": {
        "INVESTOR": (
            "MATCH (investor:Node {id: $entity_id})-[r:INVESTS_IN]->(target:Node)\n"
            "RETURN investor.label AS investor_name, investor.type AS investor_type, "
            "type(r) AS relationship, target.label AS target_name, target.type AS target_type"
        ),
        "SECTOR": (
            "MATCH (investor:Node)-[r:INVESTS_IN]->(target:Node {id: $entity_id})\n"
            "RETURN investor.label AS investor_name, investor.type AS investor_type, "
            "type(r) AS relationship, target.label AS target_name, target.type AS target_type"
        ),
        "COMPANY": (
            "MATCH (investor:Node)-[r:INVESTS_IN]->(target:Node {id: $entity_id})\n"
            "RETURN investor.label AS investor_name, investor.type AS investor_type, "
            "type(r) AS relationship, target.label AS target_name, target.type AS target_type"
        ),
        "GEOGRAPHY": (
            "MATCH (investor:Node)-[r:INVESTS_IN]->(target:Node)\n"
            "WHERE target.type IN ['COMPANY', 'SECTOR', 'PRIVATE_CO']\n"
            "RETURN investor.label AS investor_name, investor.type AS investor_type, "
            "type(r) AS relationship, target.label AS target_name, target.type AS target_type\n"
            "LIMIT 20"
        ),
        "default": (
            "MATCH (investor:Node)-[r:INVESTS_IN]->(target:Node {id: $entity_id})\n"
            "RETURN investor.label AS investor_name, investor.type AS investor_type, "
            "type(r) AS relationship, target.label AS target_name, target.type AS target_type"
        ),
    },
    "T_Beneficiary": {
        "SECTOR": (
            "MATCH (sector:Node {id: $entity_id})-[:CONTAINS]->(company:Node {type: 'COMPANY'})"
            "-[:BENEFITS_FROM]->(shock:Node)\n"
            "RETURN company.id AS id, company.label AS name, company.type AS node_type, "
            "shock.label AS shock_name, sector.label AS sector_name"
        ),
        "default": (
            "MATCH (company:Node {type: 'COMPANY'})-[:BENEFITS_FROM]->(shock:Node {id: $entity_id})\n"
            "RETURN company.id AS id, company.label AS name, company.type AS node_type, "
            "shock.label AS shock_name"
        ),
    },
    "T_RegulatoryImpact": {
        "SECTOR": (
            "MATCH (sector:Node {id: $entity_id})-[:CONTAINS]->(company:Node)"
            "-[:REGULATED_BY|HURT_BY]->(policy:Node)\n"
            "RETURN company.label AS company_name, policy.label AS policy_name, "
            "policy.type AS policy_type, sector.label AS sector_name"
        ),
        "default": (
            "MATCH (company:Node {id: $entity_id})-[r:REGULATED_BY|HURT_BY|CONSTRAINED_BY]->(policy:Node)\n"
            "RETURN company.label AS company_name, type(r) AS relationship, "
            "policy.label AS policy_name, policy.type AS policy_type"
        ),
    },
    "T_ConstraintRisk": {
        "SECTOR": (
            "MATCH (sector:Node {id: $entity_id})-[:CONTAINS]->(company:Node)"
            "-[:CONSTRAINED_BY|HURT_BY|NEEDS]->(constraint:Node)\n"
            "RETURN company.label AS entity_name, constraint.label AS constraint_name, "
            "constraint.type AS constraint_type, sector.label AS sector_name"
        ),
        "default": (
            "MATCH (entity:Node {id: $entity_id})-[r:CONSTRAINED_BY|HURT_BY|NEEDS]->(constraint:Node)\n"
            "RETURN entity.label AS entity_name, type(r) AS relationship, "
            "constraint.label AS constraint_name, constraint.type AS constraint_type"
        ),
    },
    "T_EventScenario": {
        "GEOGRAPHY": (
            "MATCH (geo:Node {id: $entity_id})<-[r:HURT_BY|BENEFITS_FROM|NEEDS|SOURCED_FROM]-(impacted:Node)\n"
            "RETURN geo.label AS event_name, type(r) AS relationship, "
            "impacted.label AS impacted_name, impacted.type AS impacted_type"
        ),
        "default": (
            "MATCH (event:Node {id: $entity_id})-[r:CAUSES|HURT_BY]->(impacted:Node)\n"
            "RETURN event.label AS event_name, type(r) AS relationship, "
            "impacted.label AS impacted_name, impacted.type AS impacted_type\n"
            "ORDER BY impacted.type, impacted.label"
        ),
    },
    "T_Hedging": {
        "default": (
            "MATCH (company:Node {id: $entity_id})-[r:HEDGES|HURT_BY|BENEFITS_FROM|PRICE_LINKED_TO]->(hedge:Node)\n"
            "RETURN company.label AS company_name, type(r) AS relationship, "
            "hedge.label AS hedge_target, hedge.type AS hedge_type"
        ),
    },
    "T_PriceLinkage": {
        "default": (
            "MATCH (entity:Node {id: $entity_id})-[r:PRICE_LINKED_TO|HURT_BY|BENEFITS_FROM]->(linked:Node)\n"
            "RETURN entity.label AS entity_name, entity.type AS entity_type, type(r) AS relationship, "
            "linked.label AS linked_name, linked.type AS linked_type"
        ),
    },
    "T_SupplyDisruption": {
        "RAW_MATERIAL": (
            "MATCH (entity:Node {id: $entity_id})<-[r:NEEDS|SOURCED_FROM|SUPPLIES_TO|HURT_BY]-(company:Node)\n"
            "RETURN company.label AS entity_name, type(r) AS relationship, "
            "entity.label AS linked_name, entity.type AS linked_type"
        ),
        "default": (
            "MATCH (entity:Node {id: $entity_id})-[r:NEEDS|SOURCED_FROM|SUPPLIES_TO|HURT_BY]->(linked:Node)\n"
            "RETURN entity.label AS entity_name, type(r) AS relationship, "
            "linked.label AS linked_name, linked.type AS linked_type"
        ),
    },
    "T_OwnershipStructure": {
        "default": (
            "MATCH (parent:Node)-[r:OWNS|JV_WITH]->(child:Node)\n"
            "WHERE parent.id = $entity_id OR child.id = $entity_id\n"
            "RETURN parent.label AS parent_name, parent.type AS parent_type, "
            "type(r) AS relationship, child.label AS child_name, child.type AS child_type"
        ),
    },
}

MULTI_ENTITY_TEMPLATES = {
    "T_StockImpact": {"source_id", "target_id"},
    "T_CausalChain": {"source_id", "target_id"},
    "T_CompareExposure": {"source_id", "target_id", "shock_id"},
    "T_MacroTransmission": {"source_id", "target_id"},
}


def load_kg_templates(path: str | Path = DEFAULT_TEMPLATE_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def expand_template(
    template_label: str,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
    shock_id: str | None = None,
    commodity_id: str | None = None,
    source_geo_id: str | None = None,
    path: str | Path = DEFAULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Expand a classifier label (e.g. T_StockImpact) into a runnable Cypher spec."""
    catalog = load_kg_templates(path)
    templates = catalog["templates"]
    if template_label not in templates:
        raise KeyError(f"Unknown template label: {template_label}")

    spec = dict(templates[template_label])
    available = {
        "source_id": source_id,
        "target_id": target_id,
        "shock_id": shock_id,
        "commodity_id": commodity_id,
        "source_geo_id": source_geo_id,
    }
    params: dict[str, str] = {}
    for key in spec.get("parameters", []):
        value = available.get(key)
        if value is not None:
            params[key] = value

    missing = [p for p in spec.get("parameters", []) if p not in params]
    result = {
        "template_label": template_label,
        "template_id": spec["template_id"],
        "description": spec["description"],
        "example_query": spec.get("example_query"),
        "source_type": spec["source_type"],
        "relationship": spec["relationship"],
        "target_type": spec["target_type"],
        "cypher": spec["cypher"],
        "parameters": params,
    }
    if missing:
        result["missing_parameters"] = missing
    return result


def _pick_by_type(matched, types: set[str], exclude_ids: set[str] | None = None) -> str | None:
    exclude_ids = exclude_ids or set()
    hits = matched[
        matched["matched_type"].isin(types) & ~matched["matched_id"].isin(exclude_ids)
    ]
    if not hits.empty:
        return hits.iloc[0]["matched_id"]
    return None


def _pick_distinct(matched, types: set[str], exclude_id: str | None) -> str | None:
    exclude = {exclude_id} if exclude_id else set()
    return _pick_by_type(matched, types, exclude)


def _commodity_id(matched) -> str | None:
    return _pick_by_type(matched, {"RAW_MATERIAL", "PRODUCT"})


def _chokepoint_for_transit(geo_id: str | None) -> str:
    if geo_id in GULF_GEO_IDS or geo_id is None:
        return CHOKEPOINT_GEO_ID
    return geo_id


def _entity_type(matched, entity_id: str | None) -> str | None:
    if not entity_id or matched.empty:
        return None
    hits = matched[matched["matched_id"] == entity_id]
    if hits.empty:
        return matched.iloc[0]["matched_type"]
    return hits.iloc[0]["matched_type"]


def _needs_single_entity_cypher(
    template_label: str, param_map: dict[str, str | None], matched
) -> bool:
    if len(matched) == 1:
        return True
    required = MULTI_ENTITY_TEMPLATES.get(template_label)
    if not required:
        return False
    if template_label == "T_CompareExposure":
        has_two_companies = (
            matched[matched["matched_type"] == "COMPANY"]["matched_id"].nunique() >= 2
        )
        if not has_two_companies:
            return True
        return not param_map.get("shock_id") or not param_map.get("target_id")
    missing = [p for p in required if not param_map.get(p)]
    return bool(missing)


def _single_entity_cypher(template_label: str, entity_id: str, entity_type: str) -> tuple[str, dict[str, str]]:
    variants = SINGLE_ENTITY_CYPHER.get(template_label, {})
    cypher = variants.get(entity_type) or variants.get("default")
    if not cypher:
        raise KeyError(f"No single-entity Cypher for {template_label}/{entity_type}")
    return cypher, {"entity_id": entity_id}


def _apply_single_entity_cypher(
    expanded: dict[str, Any],
    template_label: str,
    param_map: dict[str, str | None],
    matched,
) -> dict[str, Any]:
    entity_id = (
        param_map.get("source_id")
        or param_map.get("target_id")
        or param_map.get("investor_id")
        or param_map.get("shock_id")
        or matched.iloc[0]["matched_id"]
    )
    entity_type = _entity_type(matched, entity_id) or matched.iloc[0]["matched_type"]
    cypher, parameters = _single_entity_cypher(template_label, entity_id, entity_type)
    expanded["cypher"] = cypher
    expanded["parameters"] = parameters
    expanded.pop("missing_parameters", None)
    expanded["query_mode"] = "single_entity"
    expanded["single_entity_type"] = entity_type
    return expanded


def expand_from_matches(
    template_label: str,
    matches_df,
    *,
    path: str | Path = DEFAULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Build template params from entity-resolution matches_df rows."""
    matched = (
        matches_df.dropna(subset=["matched_id"])
        .drop_duplicates(subset=["matched_id"])
        .reset_index(drop=True)
    )
    if matched.empty:
        raise ValueError("matches_df has no resolved entities")

    company_id = _pick_by_type(matched, {"COMPANY"})
    geo_id = _pick_by_type(matched, {"GEOGRAPHY"})
    event_id = _pick_by_type(matched, {"EVENT"})
    macro_id = _pick_by_type(matched, {"MACRO_VAR"})
    sector_id = _pick_by_type(matched, {"SECTOR"})
    commodity_id = _commodity_id(matched)
    shock_id = event_id or macro_id or geo_id or matched.iloc[0]["matched_id"]

    param_map: dict[str, str | None] = {
        "source_id": None,
        "target_id": None,
        "shock_id": shock_id,
        "commodity_id": commodity_id,
        "source_geo_id": geo_id if geo_id in GULF_GEO_IDS else None,
        "investor_id": None,
    }

    if template_label == "T_StockImpact":
        param_map["source_id"] = company_id or matched.iloc[0]["matched_id"]
        param_map["target_id"] = _pick_distinct(
            matched, {"GEOGRAPHY", "EVENT", "MACRO_VAR"}, param_map["source_id"]
        )
    elif template_label == "T_CausalChain":
        param_map["source_id"] = (
            event_id or geo_id or macro_id or matched.iloc[0]["matched_id"]
        )
        param_map["target_id"] = _pick_distinct(
            matched, {"COMPANY", "MACRO_VAR", "SECTOR"}, param_map["source_id"]
        ) or (matched.iloc[-1]["matched_id"] if len(matched) > 1 else None)
    elif template_label == "T_Beneficiary":
        param_map["source_id"] = (
            event_id or macro_id or geo_id or sector_id or shock_id
        )
    elif template_label == "T_CompareExposure":
        companies = matched[matched["matched_type"] == "COMPANY"]["matched_id"].tolist()
        param_map["source_id"] = companies[0] if companies else matched.iloc[0]["matched_id"]
        param_map["target_id"] = (
            companies[1] if len(companies) > 1 else _pick_distinct(matched, {"COMPANY"}, param_map["source_id"])
        )
        param_map["shock_id"] = event_id or macro_id or geo_id or shock_id
    elif template_label == "T_ConstraintRisk":
        param_map["source_id"] = (
            company_id or sector_id or event_id or matched.iloc[0]["matched_id"]
        )
    elif template_label == "T_EventScenario":
        param_map["source_id"] = event_id or geo_id or shock_id
    elif template_label == "T_Hedging":
        param_map["source_id"] = company_id or matched.iloc[0]["matched_id"]
    elif template_label == "T_InvestmentFlow":
        investor_id = _pick_by_type(matched, {"INVESTOR"})
        param_map["target_id"] = sector_id or company_id
        param_map["investor_id"] = investor_id
        if not param_map["target_id"]:
            param_map["target_id"] = (
                geo_id or investor_id or matched.iloc[0]["matched_id"]
            )
    elif template_label == "T_MacroTransmission":
        param_map["source_id"] = macro_id or event_id or geo_id or _pick_distinct(
            matched, {"MACRO_VAR", "EVENT", "GEOGRAPHY"}, company_id
        )
        param_map["target_id"] = company_id or _pick_distinct(
            matched, {"COMPANY"}, param_map["source_id"]
        )
    elif template_label == "T_OwnershipStructure":
        param_map["source_id"] = company_id or matched.iloc[0]["matched_id"]
    elif template_label == "T_PriceLinkage":
        param_map["source_id"] = company_id or matched.iloc[0]["matched_id"]
    elif template_label == "T_RegulatoryImpact":
        param_map["source_id"] = company_id or sector_id or matched.iloc[0]["matched_id"]
    elif template_label == "T_SectorExposure":
        param_map["source_id"] = sector_id or company_id or matched.iloc[0]["matched_id"]
    elif template_label == "T_SupplyDisruption":
        param_map["source_id"] = (
            company_id or commodity_id or matched.iloc[0]["matched_id"]
        )
    elif template_label == "T_TransitRisk":
        param_map["target_id"] = _chokepoint_for_transit(geo_id)
        param_map["source_geo_id"] = geo_id if geo_id in GULF_GEO_IDS else None
    else:
        param_map["source_id"] = matched.iloc[0]["matched_id"]
        param_map["target_id"] = (
            matched.iloc[1]["matched_id"] if len(matched) > 1 else None
        )

    if (
        param_map["source_id"]
        and param_map["target_id"]
        and param_map["source_id"] == param_map["target_id"]
    ):
        alt = _pick_distinct(matched, set(matched["matched_type"]), param_map["source_id"])
        if alt:
            param_map["target_id"] = alt

    expanded = expand_template(
        template_label,
        source_id=param_map["source_id"],
        target_id=param_map["target_id"],
        shock_id=param_map["shock_id"],
        commodity_id=param_map["commodity_id"],
        source_geo_id=param_map["source_geo_id"],
        path=path,
    )

    if template_label == "T_TransitRisk":
        expanded["cypher"] = (
            "MATCH (commodity:Node)-[r:TRANSITS]->(geo:Node {id: $target_id})\n"
            "WHERE ($commodity_id IS NULL OR commodity.id = $commodity_id)\n"
            "RETURN\n"
            "  commodity.label AS commodity_name,\n"
            "  commodity.type AS commodity_type,\n"
            "  type(r) AS relationship,\n"
            "  geo.label AS chokepoint,\n"
            "  r.channel AS channel\n"
            "ORDER BY commodity_name"
        )
        expanded["parameters"]["commodity_id"] = commodity_id
        if param_map["source_geo_id"]:
            expanded["parameters"]["source_geo_id"] = param_map["source_geo_id"]

    if _needs_single_entity_cypher(template_label, param_map, matched):
        expanded = _apply_single_entity_cypher(expanded, template_label, param_map, matched)

    expanded["resolved_entities"] = matched[
        ["entity", "matched_node_name", "matched_id", "matched_type"]
    ].to_dict(orient="records")
    return expanded


def execute_template(driver, expanded: dict[str, Any]) -> tuple[list, str | None]:
    """Run the primary Cypher for the expanded template."""
    if expanded.get("missing_parameters"):
        # Last resort: try single-entity mode if we have resolved entities but incomplete params.
        resolved = expanded.get("resolved_entities") or []
        if resolved:
            matched = pd.DataFrame(resolved)
            param_map = {
                "source_id": expanded.get("parameters", {}).get("source_id"),
                "target_id": expanded.get("parameters", {}).get("target_id"),
                "shock_id": expanded.get("parameters", {}).get("shock_id"),
                "investor_id": expanded.get("parameters", {}).get("investor_id"),
            }
            try:
                expanded = _apply_single_entity_cypher(
                    expanded, expanded["template_label"], param_map, matched
                )
            except KeyError:
                return [], None
        else:
            return [], None

    params = dict(expanded["parameters"])
    if (
        expanded["template_label"] in {"T_CausalChain", "T_StockImpact", "T_MacroTransmission"}
        and params.get("source_id")
        and params.get("target_id")
        and params["source_id"] == params["target_id"]
    ):
        return [], "same_source_target"

    with driver.session() as s:
        rows = list(s.run(expanded["cypher"], **params))
    if rows:
        strategy = "single_entity" if expanded.get("query_mode") == "single_entity" else "primary"
        return rows, strategy
    return [], None
