"""
trip_pipeline.py  –  Rule-based tagging of Equibase/Brisnet trip comments
Author : <you>
Date   : 2025-05-19
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz, process


###########################################################################
# 1 ── DICTIONARIES  (trim or extend as you like) ─────────────────────────
###########################################################################

# --- Category buckets ----------------------------------------------------
CATEGORY_DICT: Dict[str, Set[str]] = {
    "starting_trouble": {
        "attempted to wheel", "bobbled", "broke in air", "broke in tangle",
        "broke slowly", "off slowly", "broke through gate", "dwelt",
        "flipped in gate", "hit gate", "lunged start", "pinched back (start)",
        "shuffled back (start)", "squeezed (start)", "stumbled",
        "unprepared start", "wheeled gate", "refused to break",
        "unruly gate", "wouldn’t load gate",
    },
    "position_trouble": {
        "altered course", "angled in", "angled out", "between horses",
        "blocked", "boxed", "carried out", "checked", "checked repeatedly",
        "circled field", "forced out", "forced wide", "in tight",
        "pinched back", "lost ground", "rough trip", "shut off",
        "steadied", "taken up", "swung wide", "wide early",
    },
    "contact_trouble": {
        "brushed", "bumped", "clipped heels", "fell", "hit rail",
        "hit with rival’s whip", "savaged",
    },
    "behavioral_issues": {
        "bolted", "bore in", "bore out", "bucked", "drifted", "erratic",
        "greenly", "fractious in gate", "fractious post parade",
        "ran off post parade", "washy post parade", "jumped tracks",
        "jumped shadow", "lugged in", "lugged out", "rank", "savaged",
        "swerved",
    },
    "equipment_issues": {
        "broken equipment", "fixed shoe", "lost irons", "lost whip",
        "lost jockey", "saddle slipped",
    },
    # … add more buckets or merge positive / negative trip comments here …
}

# --- Severity buckets (worst → best) -------------------------------------
SEVERITY_ORDER: List[str] = [
    "negative_severe",
    "negative_moderate",
    "negative_slight",
    "neutral",
    "positive_slight",
    "positive_moderate",
    "positive_severe",
]

# Map severity → phrases
SEVERITY_DICT: Dict[str, Set[str]] = {
    # Only a *subset* shown for brevity – extend as needed
    "negative_severe": {
        "bled", "bolted", "clipped heels", "distanced", "fell", "lost jockey",
        "pulled up", "returned lame", "saddle slipped", "stopped",
    },
    "negative_moderate": {
        "bumped", "checked", "dwelt", "drifted", "hard used", "hung",
        "in tight", "lugged in", "lugged out", "off slowly", "rank",
        "rough trip", "steadied", "swung wide", "weakened",
    },
    "negative_slight": {
        "altered course", "angled in", "angled out", "brief speed",
        "carried out", "empty", "evenly", "far back", "lost whip",
        "mild bid", "mild rally",
    },
    "neutral": {
        "fast pace", "inside", "pressed pace", "set pace", "unhurried early",
    },
    "positive_slight": {
        "allowed to settle", "good position", "saved ground",
        "split horses", "up for place",
    },
    "positive_moderate": {
        "closed fast", "finished well", "late rally", "rallied", "loomed boldly",
    },
    "positive_severe": {
        "easily", "handily", "much the best", "drew off", "led throughout",
    },
}

# Numeric weight (net_score)
SCORE_MAP = {
    "positive_severe":   3,
    "positive_moderate": 2,
    "positive_slight":   1,
    "neutral":           0,
    "negative_slight":  -1,
    "negative_moderate": -2,
    "negative_severe":  -3,
}


###########################################################################
# 2 ── MATCHER BUILDING  (spaCy PhraseMatcher + RapidFuzz helper) ─────────
###########################################################################

NLP = spacy.blank("en")            # Just tokeniser – no model weights
PHRASE_ATTR = "LOWER"              # Case-insensitive exact matches


def build_phrase_matcher(label2phrases: Dict[str, Set[str]]) -> PhraseMatcher:
    """Return a spaCy PhraseMatcher keyed by label."""
    m = PhraseMatcher(NLP.vocab, attr=PHRASE_ATTR)
    for label, phrases in label2phrases.items():
        patterns = [NLP.make_doc(p) for p in sorted(phrases)]
        m.add(label, patterns)
    return m


CATEGORY_MATCHER = build_phrase_matcher(CATEGORY_DICT)
SEVERITY_MATCHER = build_phrase_matcher(SEVERITY_DICT)


def fuzzy_hit(text: str, phrase: str, *, threshold: int = 90) -> bool:
    """
    True if RapidFuzz token_set_ratio ≥ threshold and phrase has ≥2 tokens.
    Prevents 'rank' matching 'frank', etc.
    """
    return (
        len(phrase.split()) >= 2
        and fuzz.token_set_ratio(text, phrase) >= threshold
    )


###########################################################################
# 3 ── CLEAN / NORMALISE  ─────────────────────────────────────────────────
###########################################################################

_TAB_RE = re.compile(r"\t+")
_WS_RE = re.compile(r"\s+")


def clean_comment(raw: str | float) -> str:
    """Lower-case, replace TAB with space, collapse whitespace/punct."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    txt = _TAB_RE.sub(" ", raw.lower())
    txt = _WS_RE.sub(" ", txt)
    return txt.strip()


###########################################################################
# 4 ── TAGGING FUNCTIONS  ────────────────────────────────────────────────
###########################################################################

def tag_categories(text: str) -> Set[str]:
    """Return 0-N category labels."""
    doc = NLP(text)
    hits = {NLP.vocab.strings[match_id] for match_id, _, _ in CATEGORY_MATCHER(doc)}
    if hits:
        return hits  # exact matches good enough here
    # Optional fuzzy back-fill (rare)
    for label, phrases in CATEGORY_DICT.items():
        if any(fuzzy_hit(text, p) for p in phrases):
            hits.add(label)
    return hits or {"uncategorized"}


# Precompute severity regexes for fallback speed (1k rows ⇒ negligible)
import re as _re

_SEV_FUZZ_CACHE = {
    label: {p for p in phrases if len(p.split()) >= 2}  # ≥2 tokens only
    for label, phrases in SEVERITY_DICT.items()
}

def tag_severities(text: str) -> List[str]:
    """Return *all* severity labels that fire (exact + fuzzy)."""
    doc = NLP(text)
    tags = [NLP.vocab.strings[match_id] for match_id, _, _ in SEVERITY_MATCHER(doc)]

    if tags:
        return tags

    # Fuzzy back-fill
    for label, phrases in _SEV_FUZZ_CACHE.items():
        if any(fuzzy_hit(text, p) for p in phrases):
            tags.append(label)
    return tags or ["neutral"]


def resolve_severity(tags: List[str]) -> Tuple[str, str, int]:
    """
    Pick worst negative AND best positive.
    Return (worst_neg, best_pos, net_score).
    """
    worst = next((s for s in SEVERITY_ORDER if s in tags and s.startswith("negative")), None)
    best  = next((s for s in reversed(SEVERITY_ORDER)
                  if s in tags and s.startswith("positive")), None)
    net   = SCORE_MAP.get(best, 0) + SCORE_MAP.get(worst, 0)
    return worst or "none", best or "none", net


###########################################################################
# 5 ── MAIN PIPELINE FUNCTION ────────────────────────────────────────────
###########################################################################

def process_trip_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame that already contains two columns:
        'pp_ext_start_comment', 'pp_trip_comment'

    Returns the same DataFrame with:
        cleaned 'all_comments',
        list columns 'categories', 'severity_tags',
        scalar columns 'worst_neg', 'best_pos', 'net_score'.
    """
    assert {"pp_ext_start_comment", "pp_trip_comment"}.issubset(df.columns), \
        "DataFrame missing required comment columns."

    # 1) clean & concat
    df = df.copy()
    df["all_comments"] = (
        df["pp_ext_start_comment"].fillna("").apply(clean_comment) + " " +
        df["pp_trip_comment"].fillna("").apply(clean_comment)
    ).str.strip()

    # 2) tagging
    df["categories"]     = df["all_comments"].apply(tag_categories)
    df["severity_tags"]  = df["all_comments"].apply(tag_severities)

    # 3) resolve
    resolved = df["severity_tags"].apply(resolve_severity)
    df[["worst_neg", "best_pos", "net_score"]] = pd.DataFrame(
        resolved.tolist(), index=df.index
    )

    return df


###########################################################################
# 6 ── AGGREGATION (“day-of-card” KPIs) ─────────────────────────────────––
###########################################################################

# ─── replace the old aggregate_day with this one ──────────────────────────
def aggregate_day(
    df: pd.DataFrame,
    id_cols: List[str] = ("race", "horse_name"),
) -> pd.DataFrame:
    """
    Aggregate per horse (or whatever id_cols you pass) and return:

        sum_net_score  – numeric sum of +/- scores
        worst_neg      – worst negative severity tag in the group (or 'none')
        best_pos       – best positive severity tag in the group (or 'none')
        trouble_tags   – semicolon-joined union of all category tags
    """
    # helper functions that operate on a *Series of lists*
    def _worst_neg(series) -> str:
        worst_tag, worst_val = "none", 0
        for tag_list in series:
            for t in tag_list:
                if t.startswith("negative") and SCORE_MAP[t] < worst_val:
                    worst_tag, worst_val = t, SCORE_MAP[t]
        return worst_tag

    def _best_pos(series) -> str:
        best_tag, best_val = "none", 0
        for tag_list in series:
            for t in tag_list:
                if t.startswith("positive") and SCORE_MAP[t] > best_val:
                    best_tag, best_val = t, SCORE_MAP[t]
        return best_tag

    def _union_categories(series) -> str:
        union: Set[str] = set()
        for cat_set in series:
            union |= cat_set
        return ";".join(sorted(union))

    agg = (
        df.groupby(list(id_cols))
          .agg(
              sum_net_score=("net_score", "sum"),
              worst_neg=("severity_tags", _worst_neg),
              best_pos=("severity_tags", _best_pos),
              trouble_tags=("categories", _union_categories),
          )
          .reset_index()
    )
    return agg



###########################################################################
# 7 ── SCRIPT ENTRY-POINT  (example usage) ────────────────────────────────
###########################################################################

if __name__ == "__main__":
    DATA_PATH = Path("past_starts_long_format.parquet")
    OUT_CSV   = Path("trip_comments_tagged.csv")
    AGG_CSV   = Path("trip_comments_agg.csv")

    if not DATA_PATH.exists():
        raise SystemExit(f"❗ Parquet file not found: {DATA_PATH}")

    # Load minimal columns
    base_cols = ["track", "race", "horse_name",
                 "pp_ext_start_comment", "pp_trip_comment"]
    df_raw = pd.read_parquet(DATA_PATH, columns=base_cols)

    # Tag & score
    df_tagged = process_trip_comments(df_raw)

    # Save full detail for audit
    df_tagged.to_csv(OUT_CSV, index=False)
    print(f"🔸 Tagged rows saved to {OUT_CSV}  ({len(df_tagged):,} rows)")

    # Aggregate per horse for handicapping
    df_agg = aggregate_day(df_tagged)
    df_agg.to_csv(AGG_CSV, index=False)
    print(f"🔹 Aggregated KPIs saved to {AGG_CSV}  ({len(df_agg):,} horses)")

    # Quick sanity peek
    print("\nTop 8 tagged rows:")
    print(df_tagged[["race", "horse_name", "all_comments",
                     "worst_neg", "best_pos", "net_score"]].head(8).to_markdown())
