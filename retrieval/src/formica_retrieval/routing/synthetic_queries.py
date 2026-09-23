"""Synthetic labelled question generator for the Formica template classes.

Why this exists
---------------
Formica et al. (2023) train their template classifier on CSQA, whose questions
carry a genuine gold class in the dataset's own `description` field. This project
has no such field: the 993-query benchmark CSV holds only `text` and
`ground_truth`. The previous pipeline papered over that by deriving "gold" labels
from `rule_label_formica()` -- the same function the classifier is trained to
reproduce AND the same function that overrides it at inference. Template accuracy
came out at 962/962 because it was comparing a function against itself.

Generating questions FROM a template fixes that at the root: the class is known by
construction, before any classifier or rule sees the text. That yields real
supervision, and it fixes the rare-class problem at the same time -- F_GroupAggregate
had 4 real examples and 15 of the 24 classes had none at all, which is why the
trained model could only ever reproduce keyword rules.

Design constraints
------------------
1. Frames must not be transliterations of RELATION_KEYWORDS or of
   rule_label_formica's `has(...)` checks. If the generator only emits the exact
   trigger words the rules look for, the classifier re-learns the rules and the
   tautology comes back one level down. Each class therefore carries several
   surface realisations, including phrasings that carry the class's MEANING
   without its stereotypical keyword.
2. The distinctions the paper found decisive -- and/or, more/less, atleast/atmost,
   exactly/approximately -- are genuinely how English marks these classes, so
   overlap with those words is expected and correct. What must vary is everything
   around them: clause order, question word, voice, and whether the comparison is
   stated as a superlative, a threshold, or a relation.
3. Vocabulary is drawn from the real graph (companies, segments, metrics) so PoS
   patterns match the benchmark's, not generic CSQA entity types.
"""

from __future__ import annotations

import random
from typing import Iterator

from .template_classes import COMPANY_SEGMENT_FALLBACK

COMPANIES = sorted(COMPANY_SEGMENT_FALLBACK)
SEGMENTS = ["Public Sector Banks", "Private Sector Banks", "Small Finance Banks"]

METRICS = [
    "net interest margin", "CASA ratio", "gross NPA ratio", "net NPA ratio",
    "return on assets", "return on equity", "capital adequacy ratio", "CET1 ratio",
    "cost-to-income ratio", "provision coverage ratio", "credit-deposit ratio",
    "net profit", "total advances", "total deposits", "operating profit",
    "slippage ratio", "book value per share", "earnings per share",
]
COUNTABLES = [
    "branches", "ATMs", "employees", "subsidiaries", "digital products",
    "overseas offices", "board members", "credit rating agencies covering it",
]
ATTRIBUTES = [
    "ticker symbol", "listing exchange", "registered head office", "legal name",
    "date of incorporation", "managing director", "chairperson", "auditor",
]
TOPICS = [
    "credit risk", "cyber security risk", "regulatory risk", "interest-rate risk",
    "retail lending", "corporate lending", "treasury operations", "microfinance",
]
GEOS = ["India", "the UAE", "Singapore", "the United Kingdom", "Hong Kong", "Bahrain"]
NUMBERS = [2, 3, 5, 10, 25, 50, 100, 500, 1000, 3000, 5000]


# Each class maps to surface frames. Slots: {c1} {c2} company, {m}/{m2} metric,
# {k} countable, {a} attribute, {t}/{t2} topic, {g}/{g2} geography, {s1}/{s2}
# segment, {n} number.
FRAMES: dict[str, list[str]] = {
    # ---- Logical (paper Table 2) ----
    "F_LogUnion": [
        "Which {m} figures are disclosed by {c1} or {c2}?",
        "Does {c1} or {c2} report a higher-quality {m}?",
        "What {a} is recorded for {c1} or {c2}?",
        "Which banks are exposed to {t} or {t2}?",
        "Is {c1} present in {g} or {g2}?",
        "Name any bank facing {t} or {t2} pressure.",
        "Which lenders disclose either {m} or {m2}?",
        "{c1} or {c2} -- which one discloses its {m}?",
    ],
    "F_LogIntersection": [
        "Which {m} figures are reported by both {c1} and {c2}?",
        "What risks affect {c1} as well as {c2}?",
        "Which banks operate in {g} and also in {g2}?",
        "Do {c1} and {c2} both disclose their {m}?",
        "Which lenders face {t} together with {t2}?",
        "Name the banks present in both {g} and {g2}.",
        "What is common to {c1} and {c2} in terms of {m}?",
    ],
    "F_LogDifference": [
        "Which {m} figures does {c1} report but not {c2}?",
        "Which banks operate in {g} but not in {g2}?",
        "What does {c1} disclose that {c2} does not?",
        "Which lenders face {t} excluding those exposed to {t2}?",
        "List banks in {g} other than {c1}.",
        "Which banks report {m} without reporting {m2}?",
        "Name lenders exposed to {t} apart from {c1}.",
        "Which banks disclose {m} yet omit {m2}?",
        "Show banks in {g}, leaving out those in {g2}.",
        "Which institutions cover {t} but skip {t2}?",
        "Find banks reporting {m} minus those reporting {m2}.",
        "Who operates in {g} while staying out of {g2}?",
    ],
    # ---- Comparative (paper Table 3) ----
    "F_CompMore": [
        "Which banks have a higher {m} than {c1}?",
        "Compare the {m} of {c1} and {c2}. Which is stronger?",
        "Does {c1} outperform {c2} on {m}?",
        "Which lenders exceed {c1} on {m}?",
        "Is {c1}'s {m} greater than {c2}'s?",
        "{c1} versus {c2} on {m} -- who leads?",
        "Name the banks whose {m} beats that of {c1}.",
    ],
    "F_CompLess": [
        "Which banks have a lower {m} than {c1}?",
        "Compare the {m} of {c1} and {c2}. Which is weaker?",
        "Does {c1} trail {c2} on {m}?",
        "Which lenders fall below {c1} on {m}?",
        "Is {c1}'s {m} smaller than {c2}'s?",
        "Name the banks whose {m} is weaker than {c1}'s.",
    ],
    "F_CompApprox": [
        "Which banks have roughly the same {m} as {c1}?",
        "Is {c1}'s {m} approximately equal to {c2}'s?",
        "Which lenders report a {m} comparable to {c1}?",
        "Does {c1} sit at about the same {m} as {c2}?",
        "Name banks with a {m} similar to {c1}'s.",
    ],
    # The Count-over-Comparative classes need the counting cue spread across many
    # surface forms ("how many", "what number of", "count", "tally", "total") --
    # with only the "how many" phrasings in training, a held-out "what number of"
    # frame collapses into the plain comparative class.
    "F_CompCountMore": [
        "How many banks have a higher {m} than {c1}?",
        "Count the lenders that exceed {c1} on {m}.",
        "What number of banks outperform {c1} on {m}?",
        "How many institutions beat {c2} on {m}?",
        "Give me the number of banks ahead of {c1} on {m}.",
        "Tally the lenders reporting a stronger {m} than {c1}.",
        "How many banks rank above {c1} by {m}?",
        "What is the count of institutions above {c1} on {m}?",
        "Total up the banks whose {m} exceeds {c1}'s.",
        "How many lenders post a bigger {m} than {c2}?",
    ],
    "F_CompCountLess": [
        "How many banks have a lower {m} than {c1}?",
        "Count the lenders that fall below {c1} on {m}.",
        "What number of banks trail {c1} on {m}?",
        "How many institutions report a weaker {m} than {c2}?",
        "Give me the number of banks behind {c1} on {m}.",
        "Tally the lenders reporting a smaller {m} than {c1}.",
        "How many banks rank below {c1} by {m}?",
        "What is the count of institutions under {c1} on {m}?",
        "Total up the banks whose {m} falls short of {c1}'s.",
        "How many lenders post a lesser {m} than {c2}?",
    ],
    "F_CompCountApprox": [
        "How many banks have roughly the same {m} as {c1}?",
        "Count the lenders reporting a {m} close to {c1}'s.",
        "What number of banks sit at approximately {c1}'s {m}?",
        "Give me the number of banks near {c1} on {m}.",
        "Tally the lenders whose {m} is about equal to {c1}'s.",
        "How many institutions match {c1} approximately on {m}?",
        "What is the count of banks around {c1}'s {m} level?",
        "Total up the lenders comparable to {c1} on {m}.",
    ],
    # ---- Quantitative (paper Table 4) ----
    "F_QuantCount": [
        "How many {k} does {c1} operate?",
        "What is the total number of {k} at {c1}?",
        "Count the {k} reported by {c1}.",
        "How many {k} did {c1} disclose?",
    ],
    "F_QuantCountAtleast": [
        "How many banks operate at least {n} {k}?",
        "Count the lenders with a minimum of {n} {k}.",
        "What number of banks run {n} or more {k}?",
        "Tally the institutions holding {n} {k} or above.",
        "Give me the number of banks with no fewer than {n} {k}.",
        "How many lenders clear the {n} {k} bar?",
    ],
    "F_QuantCountAtmost": [
        "How many banks operate at most {n} {k}?",
        "Count the lenders with no more than {n} {k}.",
        "What number of banks run {n} or fewer {k}?",
        "Tally the institutions holding {n} {k} or below.",
        "Give me the number of banks capped at {n} {k}.",
        "How many lenders stay under the {n} {k} mark?",
    ],
    "F_QuantCountApprox": [
        "How many banks operate approximately {n} {k}?",
        "Count the lenders with around {n} {k}.",
        "What number of banks run roughly {n} {k}?",
        "Tally the institutions holding about {n} {k}.",
        "Give me the number of banks with close to {n} {k}.",
        "How many lenders sit near {n} {k}?",
    ],
    "F_QuantCountEqual": [
        "How many banks operate exactly {n} {k}?",
        "Count the lenders with precisely {n} {k}.",
        "What number of banks report exactly {n} {k}?",
        "Tally the institutions holding {n} {k} exactly.",
        "Give me the number of banks with {n} {k}, no more and no fewer.",
        "How many lenders report {n} {k} on the dot?",
    ],
    "F_QuantAtleast": [
        "Which banks operate at least {n} {k}?",
        "Name the lenders with a minimum of {n} {k}.",
        "List banks running {n} or more {k}.",
        "Which institutions hold at least {n} {k}?",
    ],
    "F_QuantAtmost": [
        "Which banks operate at most {n} {k}?",
        "Name the lenders with no more than {n} {k}.",
        "List banks running {n} or fewer {k}.",
        "Which institutions hold at most {n} {k}?",
    ],
    "F_QuantApprox": [
        "Which banks operate approximately {n} {k}?",
        "Name the lenders with around {n} {k}.",
        "List banks running roughly {n} {k}.",
        "Which institutions hold about {n} {k}?",
        "Find banks with close to {n} {k}.",
        "Who runs in the region of {n} {k}?",
    ],
    "F_QuantEqual": [
        "Which banks operate exactly {n} {k}?",
        "Name the lenders with precisely {n} {k}.",
        "List banks running exactly {n} {k}.",
        "Which institutions hold {n} {k}, no more and no fewer?",
        "Find banks with a count of {n} {k} exactly.",
        "Who reports {n} {k} on the dot?",
    ],
    "F_QuantMax": [
        "Which bank operates the maximum number of {k}?",
        "Name the lender with the largest {m}.",
        "Which institution tops the table on {m}?",
        "Who runs the greatest number of {k}?",
        "Identify the bank with the highest {m}.",
        "Which lender reports the peak {m}?",
        "What bank holds the most {k}?",
        "Point me to the institution leading on {m}.",
        "Which bank maximises {m}?",
        "Whose {m} is the biggest?",
    ],
    "F_QuantMin": [
        "Which bank operates the minimum number of {k}?",
        "Name the lender with the smallest {m}.",
        "Which institution sits at the bottom on {m}?",
        "Who runs the fewest {k}?",
        "Identify the bank with the lowest {m}.",
        "Which lender reports the floor {m}?",
        "What bank holds the least {k}?",
        "Point me to the institution lagging on {m}.",
        "Which bank minimises {m}?",
        "Whose {m} is the smallest?",
    ],
    # ---- Simple (paper Table 5) ----
    "F_Simple": [
        "What is {c1}'s {m}?",
        "What is the {a} of {c1}?",
        "Where is {c1} headquartered?",
        "Who is the {a} of {c1}?",
        "Describe {c1}'s exposure to {t}.",
        "What {m} did {c1} report?",
        "Tell me about {c1}'s {t} position.",
        "When was {c1} incorporated?",
        "Which products does {c1} offer?",
        "What is {c1}'s strategy in {g}?",
    ],
    # ---- Banking extensions beyond the paper's 21 ----
    "F_GlobalRank": [
        "Across all listed Indian banks, which reports the highest {m}?",
        "Among every bank in the dataset, who has the lowest {m}?",
        "Which {s1} bank leads on {m}?",
        "Of all the lenders covered, which posts the strongest {m}?",
        "Rank the banks by {m} and name the top one.",
        "Which {s1} entity reports the weakest {m}?",
        "Over the whole universe of banks, who tops {m}?",
        "Out of all 39 listed banks, which has the best {m}?",
        "Considering every {s1} institution, which ranks first on {m}?",
        "Sort all lenders by {m} -- who comes last?",
        "Economy-wide, which bank reports the strongest {m}?",
    ],
    "F_GroupAggregate": [
        "How do {s1} as a group compare to {s2} on {m}?",
        "Which segment has the highest average {m}?",
        "Compare {s1} and {s2} on average {m}.",
        "Is the mean {m} higher for {s1} or for {s2}?",
        "Which category of banks averages the lowest {m}?",
        "Collectively, do {s1} beat {s2} on {m}?",
        "Which ownership group reports the better mean {m}?",
        "Aggregate {m} by segment -- which leads?",
        "Taken together, how does {s1} fare against {s2} on {m}?",
        "Across segments, where is average {m} strongest?",
        "Which bank category posts the highest mean {m}?",
    ],
    "F_CompToGroupAverage": [
        "What is {c1}'s {m}, and how does it compare with the average for its segment?",
        "Is {c1}'s {m} above or below its peer-group average?",
        "How does {c1}'s {m} stand against the segment average?",
        "Compare {c1}'s {m} to the average of its peer group.",
        "Does {c1} beat the typical {m} for banks like it?",
        "Where does {c1}'s {m} sit relative to its category mean?",
        "Benchmark {c1}'s {m} against its own segment.",
        "Is {c1} ahead of or behind the average bank in its group on {m}?",
        "How far is {c1}'s {m} from its peer average?",
    ],
}


def _fill(frame: str, rng: random.Random) -> str:
    c1, c2 = rng.sample(COMPANIES, 2)
    m, m2 = rng.sample(METRICS, 2)
    t, t2 = rng.sample(TOPICS, 2)
    g, g2 = rng.sample(GEOS, 2)
    s1, s2 = rng.sample(SEGMENTS, 2)
    return frame.format(
        c1=c1, c2=c2, m=m, m2=m2, t=t, t2=t2, g=g, g2=g2, s1=s1, s2=s2,
        k=rng.choice(COUNTABLES), a=rng.choice(ATTRIBUTES), n=rng.choice(NUMBERS),
    )


def generate(
    per_class: int = 200,
    seed: int = 13,
    frame_slice: str = "all",
    holdout: float = 0.3,
) -> Iterator[tuple[str, str]]:
    """Yield (question, label) pairs, balanced across every template class.

    Balanced on purpose: the real benchmark is 63% F_Simple, and training on that
    distribution is what taught the previous model to answer F_Simple whenever it
    was unsure -- the exact failure that forced the inference-time rule overrides.

    `frame_slice` controls FRAME-level splitting, which is the only split that
    measures anything real. A random split over generated questions puts the same
    frame in train and test with different companies substituted in, so a model
    that memorises frames scores 100% while having learned nothing transferable --
    and the benchmark's real questions are phrased by a human, not by these
    frames. Training on "train" frames and testing on "holdout" frames asks the
    question that actually matters: does it generalise to phrasings it has never
    seen?
    """
    rng = random.Random(seed)
    for label, frames in FRAMES.items():
        pool = sorted(frames)
        if frame_slice != "all":
            split_rng = random.Random(f"{seed}-{label}")
            shuffled = pool[:]
            split_rng.shuffle(shuffled)
            n_holdout = max(1, int(round(len(shuffled) * holdout)))
            pool = shuffled[n_holdout:] if frame_slice == "train" else shuffled[:n_holdout]
        if not pool:
            continue
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_class and attempts < per_class * 40:
            attempts += 1
            text = _fill(rng.choice(pool), rng)
            if text not in seen:
                seen.add(text)
                yield text, label


def class_coverage() -> dict[str, int]:
    """Frames defined per class -- a class with too few cannot vary its phrasing."""
    return {label: len(frames) for label, frames in FRAMES.items()}
