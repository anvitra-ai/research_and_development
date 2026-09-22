"""KG-row -> natural-language summary: hop-path text formatting + Gemini prompt.

Two-step process used by pipeline.process_query():
  1. format_kg_rows_for_summary() turns raw Neo4j result rows into readable text
     (preferring Graphiti's own stored r.fact over reconstructing a sentence).
  2. summarize_hops() sends that text to Gemini with a fixed set of answer-quality
     requirements and returns the plain-language answer.
"""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types

from . import config as _cfg

# "Best reasoning model" per graphiti/run_ingestion.py's own designation, used
# there for extraction. Swapped in here from gemini-3.5-flash-lite because the
# flash-lite tier is the dominant cause of the "fetched but missed" failure
# class: 24 of 175 Not-relevant v16 queries had every material fact already in
# kg_hop_path and the summary still said "not covered" or dropped a figure that
# was sitting right there in the rows. That is a reasoning/attention failure on
# text already in context, which a stronger model is squarely suited to fix --
# unlike the extraction_gap and no-company buckets, this one needed no new
# retrieval logic, just a model actually capable of reading its own input.
SUMMARY_MODEL = _cfg.SUMMARY_MODEL

# GeminiClient's max_tokens default keys off the model name and doesn't
# recognise "gemini-3.1-pro-preview" (see the identical note in
# graphiti/run_ingestion.py), silently capping output at 8192 tokens. Summaries
# are short, but the note is left here because it is exactly the kind of thing
# that fails silently and was hard to diagnose the first time.
SUMMARY_MAX_TOKENS = _cfg.SUMMARY_MAX_TOKENS

# Greedy decoding, for the same reason the judge uses it (see llm_judge.
# JUDGE_TEMPERATURE): the summariser is the OTHER half of this benchmark's
# run-to-run noise. Measured directly -- re-running queries whose retrieval was
# byte-identical (same strategy, same single row) still flipped their judged
# label, purely because the summary was sampled differently. With both ends
# greedy, a change in score is attributable to a change in the pipeline.
SUMMARY_TEMPERATURE = _cfg.SUMMARY_TEMPERATURE


def _format_date(value: Any) -> str | None:
    """Render a Neo4j DateTime as YYYY-MM-DD, but only when it's a genuine reported
    date and not Graphiti's ingestion-time default.

    When an episode's text has no extractable date for a fact, Graphiti falls back
    to stamping valid_at with the episode's own processing wall-clock time (e.g.
    2026-09-16 07:38:12) rather than leaving it null. A real date parsed from text
    ("Q1 FY27", "March 31, 2026", ...) always normalizes to midnight; a scattered,
    non-midnight time-of-day is the tell for the ingestion-time default. Surfacing
    the latter as "as of <date>" actively misleads the summary (it reads like the
    reported-as-of date when it's really just when the episode was loaded), so
    only midnight timestamps are treated as real dates -- anything else is dropped.
    """
    if value is None:
        return None
    if hasattr(value, "to_native"):
        value = value.to_native()
    time_part = getattr(value, "time", None)
    if callable(time_part):
        t = time_part()
        if (t.hour, t.minute, t.second) != (0, 0, 0):
            return None
    if hasattr(value, "date"):
        value = value.date()
    return str(value)


def format_kg_rows_for_summary(rows: list, expanded: dict[str, Any]) -> str:
    """Turn Neo4j result rows into readable hop/path text for Gemini summarization.

    Prefers Graphiti's own stored text over reconstructing a sentence from raw
    subject/relationship/object triples, since it's already a precise, human-written
    summary of that specific fact:
      - attribute_lookup rows (a node's own property, not an edge) -> "X's <attr> is <value>"
      - a single edge's r.fact -> the fact text verbatim
      - aggregate/list facts (F_QuantCount's collect(), F_CompMore/Less/Approx's path
        facts_a/facts_b) -> the count/comparison header plus every underlying fact
    Falls through to the legacy path/transit formatting, then the generic key=value
    reconstruction, for rows that carry none of the above (e.g. the still-unfixed
    legacy :Node-schema special modes, which have no fact/attribute fields at all).
    """
    if not rows:
        return ""
    lines = [
        f"Template: {expanded['template_id']} ({expanded['template_label']})",
        f"Query intent: {expanded['description']}",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        data = dict(row)
        if data.get("profile"):
            # A node's own Graphiti-generated `summary` property, surfaced the way
            # r.fact is surfaced for edges. Worth including because extraction is
            # lossy in a DIFFERENT way per node: measured on v17's failures, the
            # node summary alone contained every missing ground-truth token for 20
            # queries and some of them for 24 more, purely because the summariser
            # writes figures into the profile that never became their own edges.
            #
            # Verified before trusting it as evidence: every number in a sampled
            # node summary also appears in that bank's source section (13 of the
            # section's 31 numbers, none invented) -- it is lossy, not inventive,
            # which is what makes it safe to quote from.
            lines.append(f"Row {i}: Profile of {data.get('subject_name', '?')}: {data['profile']}")
        elif "max_value" in data or "min_value" in data:
            # global_ranking_max/min has ALREADY ranked every company and returns
            # the single winner. Formatted as an ordinary fact, that row reads as
            # one isolated datapoint about one bank, and the summariser concluded
            # the ranking question was unanswered -- verified on v19: the row said
            # "Bank of Baroda's return on assets was 0.25%", which is exactly the
            # ground-truth answer, and the summary still said "the retrieved facts
            # do not cover which bank reports the lowest". The computation was
            # right and the presentation threw the answer away, so say plainly
            # that this row IS the ranked result.
            is_max = "max_value" in data
            value = data.get("max_value") if is_max else data.get("min_value")
            subject = data.get("subject_name", "?")
            metric = data.get("object_name") or "the requested metric"
            segment = data.get("graph_segment")
            scope = f" among {segment}s" if segment else " across all companies compared"
            lines.append(
                f"Row {i}: RANKED RESULT -- {subject} has the "
                f"{'HIGHEST' if is_max else 'LOWEST'} {metric}{scope}, at {value}. "
                f"This ranking was computed over every company that reports this metric, "
                f"so {subject} is the answer."
            )
            if data.get("fact"):
                as_of = _format_date(data.get("valid_at"))
                lines.append(f"  Supporting fact: {data['fact']}{f' (as of {as_of})' if as_of else ''}")
        elif "attribute" in data and "value" in data:
            subject = data.get("subject_name", "?")
            lines.append(f"Row {i}: {subject}'s {data['attribute']} is {data['value']}")
        elif "segment_average" in data:
            # F_CompToGroupAverage: the company's own fact PLUS its segment
            # peers' average -- both numbers matter here, so this can't just
            # fall into the generic single-fact branch below (which would print
            # the company's own fact and silently drop the average entirely).
            subject = data.get("subject_name", "?")
            seg = data.get("segment_name")
            avg = data.get("segment_average")
            peers = data.get("peer_count")
            fact = data.get("fact") or f"{subject}'s value is {data.get('value')}"
            lines.append(f"Row {i}: {fact}")
            if avg is not None:
                lines.append(
                    f"  Segment average ({seg or 'peer group'}, {peers or 0} other companies): {avg}"
                )
            else:
                lines.append(f"  No segment-average data available for {seg or 'this company'}'s peers.")
        elif data.get("fact"):
            as_of = _format_date(data.get("valid_at"))
            suffix = f" (as of {as_of})" if as_of else ""
            # Anchor with the full subject name explicitly -- fact text often uses
            # a bank's short ticker/abbreviation (e.g. "PNB"), and without the
            # unambiguous full name alongside it the summarizer can expand the
            # abbreviation to an unrelated, more globally-common institution (it
            # once turned "PNB" into "Philippine National Bank" instead of
            # "Punjab National Bank" purely because the fact text never said which).
            subject = data.get("subject_name")
            prefix = (
                f"[{subject}] " if subject and not data["fact"].lower().startswith(subject.lower()) else ""
            )
            lines.append(f"Row {i}: {prefix}{data['fact']}{suffix}")
        elif data.get("facts"):
            # F_QuantCount's collect(r.fact) used to drop collect(r.valid_at)
            # entirely, so "...and as of what date?" queries could never surface
            # a date even when every underlying edge had one -- the summarizer
            # would then either invent nothing (correct but unhelpful) or, worse,
            # claim outright that no date was given. valid_ats is genuinely
            # optional here (older cyphers/legacy :Node-schema rows won't have
            # it), so this only activates when it's actually present.
            raw_facts = data.get("facts") or []
            raw_dates = data.get("valid_ats") or [None] * len(raw_facts)
            pairs = [(f, d) for f, d in zip(raw_facts, raw_dates) if f]
            header_parts = [
                f"{k}={v}" for k, v in data.items() if v is not None and k not in ("facts", "valid_ats")
            ]
            lines.append(f"Row {i}: " + " | ".join(header_parts) + f" ({len(pairs)} facts)")
            for f, d in pairs:
                as_of = _format_date(d)
                suffix = f" (as of {as_of})" if as_of else ""
                lines.append(f"  - {f}{suffix}")
        elif data.get("facts_a") or data.get("facts_b"):
            header_parts = [
                f"{k}={v}" for k, v in data.items() if v is not None and k not in ("facts_a", "facts_b")
            ]
            lines.append(f"Row {i}: " + " | ".join(header_parts))
            for label, facts in (("a", data.get("facts_a")), ("b", data.get("facts_b"))):
                for f in facts or []:
                    if f:
                        lines.append(f"  [{label}] {f}")
        elif "path_names" in data and "rel_types" in data:
            names = data.get("path_names") or []
            rels = data.get("rel_types") or []
            header = (
                f"Path {i}: {data.get('source_name', names[0] if names else '?')}"
                f" -> {data.get('target_name', names[-1] if names else '?')}"
                f" ({data.get('hops', len(rels))} hops)"
            )
            lines.append(header)
            for j, rel in enumerate(rels):
                left = names[j] if j < len(names) else "?"
                right = names[j + 1] if j + 1 < len(names) else "?"
                lines.append(f"  {left} -[{rel}]-> {right}")
        elif "commodity_name" in data and ("geography_name" in data or "chokepoint" in data):
            geo = data.get("geography_name") or data.get("chokepoint")
            share = f" share={data['share']}" if data.get("share") is not None else ""
            extra = f" [{data['channel']}]" if data.get("channel") else ""
            lines.append(
                f"Row {i}: {data['commodity_name']} -[TRANSITS]-> {geo}{share}{extra}"
            )
        else:
            parts = [f"{k}={v}" for k, v in data.items() if v is not None]
            lines.append(f"Row {i}: " + " | ".join(parts))
    return "\n".join(lines)


def summarize_hops(client: genai.Client, hop_path: str, query: str = "") -> str:
    question_block = f'User question: "{query}"\n\n' if query else ""
    prompt = f"""
You are a financial analyst answering a business question using knowledge-graph facts.

{question_block}Write a plain-language answer to the question above, in business terms, using
only the facts below.

Requirements:
1. Answer the question directly. If the facts below include the specific figure
   or detail the question asks for, lead with it. If they do not, say plainly
   that the retrieved facts don't cover it -- do not paper over the gap by
   summarizing whatever unrelated facts happen to be present instead.
2. If the question asks to compare two companies on ONE specific named metric
   (e.g. "gross NPA ratio", "return on equity"), the facts below may list MANY
   metrics for each company, not just that one -- find the matching figure for
   each company specifically (not just any number), state both values, and do
   the arithmetic yourself: which is stronger and by how much (the numeric or
   percentage-point difference). Never say a comparison "cannot be made" when
   both companies' values for that exact metric are actually present below.
   If a fact presents a "headline"/reported figure alongside an adjusted one
   (e.g. "return on equity of 3.89% reported, or 16.57% excluding a one-off
   item"), do the comparison and the arithmetic on the HEADLINE reported
   figure, and mention the adjusted figure as a caveat alongside it. Reported
   figures are what the two banks' disclosures actually state, so they are the
   like-for-like basis; silently substituting an adjusted number changes both
   the stated value and the computed difference (e.g. answering "stronger by
   1.30 points" off 16.57% when the reported 3.89% gives 13.98 points). State
   both, compute on the headline.
3. If the question offers a disjunctive fallback ("its government promoter OR
   largest shareholder", "the Chairman OR MD & CEO, whichever is the top
   executive title"), and the facts below don't cover the FIRST option but do
   cover the SECOND (e.g. no government promoter exists for this company, but
   the facts list several institutional/retail shareholders with percentages),
   answer using the option the facts actually support -- state the largest of
   the listed holders as the answer to "largest shareholder," rather than
   declaring the whole question unanswerable just because the first-named
   option wasn't found. Only say "not covered" if NEITHER option is supported.
4. Requirement 8 below forbids inventing NUMBERS -- it does not forbid ordinary
   reasoning about numbers or facts that are actually present. Two situations
   in particular are NOT "not covered":
   - Synthesis questions ("what customer segment does X prioritize", "what is
     X's business focus") where the facts below list several related but
     separate items (e.g. "serves retail customers", "serves corporate
     customers", "serves MSME customers", branch/product details) without one
     single fact spelling out the synthesized answer. Name the segment(s) the
     facts actually support, weighting whichever the facts emphasize most
     (mentioned repeatedly, tied to the bank's core product mix, or given the
     most supporting detail) -- do not decline to answer just because no
     single fact uses the exact word "prioritize."
   - Standard-interpretation questions ("what does that imply", "what does
     that indicate/suggest") following a stated figure. Apply ordinary,
     well-established financial-analysis reasoning to the number itself (e.g.
     a HIGH provision coverage ratio indicates a stronger reserve buffer
     against future bad-loan losses; a LOW one indicates a thinner cushion) --
     this is standard interpretation of a real, present figure, not invented
     content, and is required whether or not a separate fact states the
     implication in so many words.
   Only say "not covered" when the facts genuinely lack the figure or the
   related items needed to answer -- not merely because no fact states the
   final synthesis or implication in those exact words.
   A row beginning "RANKED RESULT --" is the finished answer to a ranking or
   superlative question ("which bank has the highest/lowest X?"). The ranking
   was already computed across every company that reports the metric, so state
   that company and its value as the answer. Do NOT say the facts don't cover
   which company ranks highest/lowest -- the single row IS the result, and only
   one company appears precisely because the others were ranked and eliminated.
   A row beginning "Profile of <company>:" is a company profile drawn from the
   same source material as the other facts, and counts as evidence exactly like
   them -- read it and answer from it. It is frequently the only row carrying
   descriptive detail (business focus, regional presence, ownership history,
   headcount), so answering "not covered" while a profile row below states the
   answer is a mistake.
5. If the question asks which SOURCE is cited for a fact, a fact phrased as
   "<Source name> supports the fact that ..." or "<Source name> reported/
   provided ..." IS the direct answer -- state that source's name plainly
   (e.g. "The primary source is indianmasterminds.com's July 2026 report").
   If several different sources appear for different, unrelated facts, pick
   the one(s) tied to the specific topic the question asks about (e.g.
   "financial metrics" specifically, not an unrelated merger-history or
   leadership fact also in the list) rather than declining to answer.
6. Focus only on the business/financial substance -- never describe the graph
   itself (no "starting entity", "ending entity", "relationship", "path", or any
   other entity/graph-structure framing, and no numbered or labeled sections).
7. Write flowing prose, not a list.
8. Include specific numbers, dates, or ratios only if they appear in the facts
   below; never invent or estimate them. (This is about NUMBERS specifically --
   see requirement 4 above for reasoning about numbers/facts that ARE present.)
9. If a fact states a figure but also flags that figure as questionable, unusual,
   or inconsistent (e.g. "although this reported number appears inconsistent
   with the bank's actual scale"), still report the figure -- carry the caveat
   along with it (e.g. "a reported net interest income of X crore, though the
   source flags this as inconsistent with the bank's scale") rather than
   treating the whole fact as if it were never covered at all.
10. Maximum 2 sentences -- UNLESS the question asks to enumerate multiple
   distinct items of the same kind (e.g. "what risks is X exposed to",
   "which shareholder categories hold a stake", "what products/services does
   X offer", "what is the main weakness" when several are listed below). For
   those, use up to 4 sentences so every distinct item the facts below
   actually support gets named, rather than stopping after the first one or
   two and dropping the rest. This also applies to "who is the MD & CEO (or
   Chairman...)" leadership questions when the facts below name BOTH an MD &
   CEO/managing director AND a separate Chairman/chairperson -- state both
   roles and names, don't stop after the first one just because the question
   phrased it as an "or".

Knowledge graph facts:

{hop_path}
"""
    return _generate_with_retry(client, prompt)


# A Gemini call with no timeout blocks its thread forever if the connection
# stalls, and that is fatal under concurrency rather than merely slow: a v19 run
# wedged permanently at 530/993 with all 8 worker threads holding open HTTPS
# connections to Google that never returned. Serial runs hid the problem because
# one stalled call looks like "slow" rather than "dead". Bounding the wait and
# retrying turns an unrecoverable hang into a delay.
SUMMARY_TIMEOUT_MS = _cfg.SUMMARY_TIMEOUT_MS
SUMMARY_MAX_RETRIES = _cfg.SUMMARY_MAX_RETRIES


def _generate_with_retry(client: genai.Client, prompt: str) -> str:
    config = types.GenerateContentConfig(
        temperature=SUMMARY_TEMPERATURE,
        max_output_tokens=SUMMARY_MAX_TOKENS,
        http_options=types.HttpOptions(timeout=SUMMARY_TIMEOUT_MS),
    )
    last_error: Exception | None = None
    for attempt in range(SUMMARY_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=SUMMARY_MODEL, contents=prompt, config=config
            ).text
        except Exception as exc:  # timeouts, transient 5xx, connection resets
            last_error = exc
            if attempt == SUMMARY_MAX_RETRIES:
                break
            delay = 5 * (2**attempt)
            print(
                f"  summarization error (attempt {attempt + 1}/{SUMMARY_MAX_RETRIES + 1}): "
                f"{exc!r} -- retrying in {delay}s"
            )
            time.sleep(delay)
    # Returning empty rather than raising keeps one bad query from killing a
    # whole batch run; process_query records it as an empty summary, which the
    # judge already handles by auto-labelling "Not relevant".
    print(f"  summarization failed after {SUMMARY_MAX_RETRIES + 1} attempts: {last_error!r}")
    return ""
