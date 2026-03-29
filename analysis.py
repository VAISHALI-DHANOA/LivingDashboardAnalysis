#!/usr/bin/env python3
"""
Analysis script for HCI study comparing Living Dashboard (System A) vs Baseline (System B).
Computes SUS, NASA-TLX, satisfaction, LD-specific items, and comparative scores.
Exports results to results.json.
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# ── Configuration ──────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "Numerical_Transposed_LimeSurvey.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results.json")

SYSTEM_A_PARTICIPANTS = {"P1", "P2", "P3", "P10", "P11", "P12"}
SYSTEM_B_PARTICIPANTS = {"P4", "P5", "P6", "P7", "P8", "P9"}

# Display labels: System A → L1-L6, System B → B1-B6
PID_RENAME = {
    "P1": "L1", "P2": "L2", "P3": "L3", "P10": "L4", "P11": "L5", "P12": "L6",
    "P4": "B1", "P5": "B2", "P6": "B3", "P7": "B4", "P8": "B5", "P9": "B6",
}

# ── Keyword mappings ──────────────────────────────────────────────────────────

SUS_KEYWORDS = {
    "frequently": "SUS1 - Frequency",
    "unnecessarily complex": "SUS2 - Complexity",
    "easy to use": "SUS3 - Easy to use",
    "technical person": "SUS4 - Technical support",
    "well integrated": "SUS5 - Well integrated",
    "inconsistency": "SUS6 - Inconsistency",
    "learn to use this system very quickly": "SUS7 - Quick to learn",
    "cumbersome": "SUS8 - Cumbersome",
    "confident": "SUS9 - Confidence",
    "learn a lot of things before": "SUS10 - Learning curve",
}

NASA_KEYWORDS = {
    "mentally demanding": "Mental Demand",
    "hard did you have to work": "Effort",
    "annoyed or irritated": "Annoyance",
    "how successful": "Success",
}

LD_KEYWORDS = {
    "sensible decisions": "Sensible Decisions",
    "understood why the dashboard changed": "Understood Layout",
    "felt in control": "Felt in Control",
    "comfortable overriding": "Comfortable Overriding",
    "automated the right amount": "Right Amount",
    "shrinking of views felt appropriate": "Shrinking Appropriate",
    "visual changes (shrinking, colour shifts)": "Visual Communication",
    "way views changed appearance helped me understand": "Appearance Understanding",
    "hiding/shrinking inactive views helped me focus": "Helped Focus",
}

SATISFACTION_KEYWORD = "how satisfied"
AI_SATISFACTION_KEYWORD = "suggestions provided by the AI"
COMPARATIVE_KEYWORD = "Compared to a static dashboard"


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_column(df, keyword):
    """Find a column whose header contains the keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    matches = [c for c in df.columns if keyword_lower in c.lower()]
    if len(matches) == 0:
        sys.exit(f"ERROR: No column found for keyword '{keyword}'")
    if len(matches) > 1:
        # Pick the best match (shortest column name = most specific)
        matches.sort(key=len)
    return matches[0]


def condition(pid):
    """Return 'A' or 'B' based on participant ID."""
    if pid in SYSTEM_A_PARTICIPANTS:
        return "A"
    elif pid in SYSTEM_B_PARTICIPANTS:
        return "B"
    else:
        sys.exit(f"ERROR: Unknown participant '{pid}'")


def stats(values):
    """Return dict with mean and sd."""
    arr = np.array(values, dtype=float)
    return {"mean": round(float(np.mean(arr)), 2), "sd": round(float(np.std(arr, ddof=1)), 2)}


# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# Identify participant column
pid_col = [c for c in df.columns if "participant" in c.lower()][0]
df[pid_col] = df[pid_col].astype(str).str.strip()
df = df.set_index(pid_col)

print(f"Loaded {len(df)} participants: {', '.join(df.index)}")

# Assign conditions
df["condition"] = df.index.map(condition)

# Rename participant IDs for display
df.index = df.index.map(lambda pid: PID_RENAME.get(pid, pid))

# ── Resolve all columns by keyword ────────────────────────────────────────────

sus_cols = {label: find_column(df, kw) for kw, label in SUS_KEYWORDS.items()}
nasa_cols = {label: find_column(df, kw) for kw, label in NASA_KEYWORDS.items()}
ld_cols = {label: find_column(df, kw) for kw, label in LD_KEYWORDS.items()}
satisfaction_col = find_column(df, SATISFACTION_KEYWORD)
ai_satisfaction_col = find_column(df, AI_SATISFACTION_KEYWORD)
comparative_col = find_column(df, COMPARATIVE_KEYWORD)

# Qualitative columns = last three original columns (before we added 'condition')
original_cols = [c for c in df.columns if c != "condition"]
qual_cols = original_cols[-3:]

print(f"\nQualitative columns:\n  1. {qual_cols[0]}\n  2. {qual_cols[1]}\n  3. {qual_cols[2]}")

# ── SUS Scoring ────────────────────────────────────────────────────────────────
# All items already reverse-encoded for negative items.
# Formula: sum((item - 1) for all 10 items) * 2.5

sus_results = {}
for pid in df.index:
    item_scores = []
    for label, col in sus_cols.items():
        val = float(df.loc[pid, col])
        item_scores.append(val - 1)
    sus_score = round(sum(item_scores) * 2.5, 2)
    sus_results[pid] = {
        "score": sus_score,
        "condition": df.loc[pid, "condition"],
    }

sus_a = [v["score"] for v in sus_results.values() if v["condition"] == "A"]
sus_b = [v["score"] for v in sus_results.values() if v["condition"] == "B"]

sus_output = {
    "participants": sus_results,
    "condition_stats": {
        "A": stats(sus_a),
        "B": stats(sus_b),
    },
}

# ── NASA-TLX ───────────────────────────────────────────────────────────────────

nasa_output = {"items": {}, "workload_aggregate": {}}

for label, col in nasa_cols.items():
    vals_a = [float(df.loc[pid, col]) for pid in df.index if df.loc[pid, "condition"] == "A"]
    vals_b = [float(df.loc[pid, col]) for pid in df.index if df.loc[pid, "condition"] == "B"]
    nasa_output["items"][label] = {"A": stats(vals_a), "B": stats(vals_b)}

# Workload aggregate: mean of the three negative items (exclude Success)
negative_items = ["Mental Demand", "Effort", "Annoyance"]
for cond, pids in [("A", SYSTEM_A_PARTICIPANTS), ("B", SYSTEM_B_PARTICIPANTS)]:
    agg_scores = []
    for pid in df.index:
        if df.loc[pid, "condition"] != cond:
            continue
        neg_vals = [float(df.loc[pid, nasa_cols[item]]) for item in negative_items]
        agg_scores.append(np.mean(neg_vals))
    nasa_output["workload_aggregate"][cond] = stats(agg_scores)

# ── Satisfaction ───────────────────────────────────────────────────────────────

sat_a = [float(df.loc[pid, satisfaction_col]) for pid in df.index if df.loc[pid, "condition"] == "A"]
sat_b = [float(df.loc[pid, satisfaction_col]) for pid in df.index if df.loc[pid, "condition"] == "B"]

ai_a = [float(df.loc[pid, ai_satisfaction_col]) for pid in df.index if df.loc[pid, "condition"] == "A"]
ai_b = [float(df.loc[pid, ai_satisfaction_col]) for pid in df.index if df.loc[pid, "condition"] == "B"]

satisfaction_output = {"A": stats(sat_a), "B": stats(sat_b)}
ai_satisfaction_output = {"A": stats(ai_a), "B": stats(ai_b)}

# ── LD-Specific Items (System A only) ─────────────────────────────────────────

ld_output = {}
for label, col in ld_cols.items():
    vals = [float(df.loc[pid, col]) for pid in df.index if df.loc[pid, "condition"] == "A"]
    ld_output[label] = stats(vals)

# ── Comparative Item (System A only) ──────────────────────────────────────────

comp_values = {}
for pid in df.index:
    if df.loc[pid, "condition"] == "A":
        comp_values[pid] = float(df.loc[pid, comparative_col])

comparative_output = {
    "values": comp_values,
    "mean": round(float(np.mean(list(comp_values.values()))), 2),
}

# ── Qualitative Feedback ──────────────────────────────────────────────────────

qualitative_output = {}
for pid in df.index:
    qualitative_output[pid] = {
        "condition": df.loc[pid, "condition"],
        "most_useful": str(df.loc[pid, qual_cols[0]]) if pd.notna(df.loc[pid, qual_cols[0]]) else "",
        "confusing_missing": str(df.loc[pid, qual_cols[1]]) if pd.notna(df.loc[pid, qual_cols[1]]) else "",
        "ld_comments": str(df.loc[pid, qual_cols[2]]) if pd.notna(df.loc[pid, qual_cols[2]]) else "",
    }

# ── Export to JSON ─────────────────────────────────────────────────────────────

results = {
    "sus": sus_output,
    "nasa_tlx": nasa_output,
    "overall_satisfaction": satisfaction_output,
    "ai_satisfaction": ai_satisfaction_output,
    "ld_specific": ld_output,
    "comparative": comparative_output,
    "qualitative": qualitative_output,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults exported to {OUTPUT_PATH}")

# ── Print Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STUDY RESULTS SUMMARY")
print("=" * 70)

print("\n── SUS Scores ──────────────────────────────────────────────────────")
print(f"{'Participant':<14} {'Condition':<12} {'SUS Score':<10}")
print("-" * 36)
for pid in sorted(sus_results.keys(), key=lambda x: int(x[1:])):
    r = sus_results[pid]
    print(f"{pid:<14} {'System ' + r['condition']:<12} {r['score']:<10.1f}")
print("-" * 36)
print(f"{'System A Mean':<14} {'':<12} {sus_output['condition_stats']['A']['mean']:.1f} (SD={sus_output['condition_stats']['A']['sd']:.1f})")
print(f"{'System B Mean':<14} {'':<12} {sus_output['condition_stats']['B']['mean']:.1f} (SD={sus_output['condition_stats']['B']['sd']:.1f})")

print("\n── NASA-TLX (1-7 scale) ────────────────────────────────────────────")
print(f"{'Item':<18} {'System A':<20} {'System B':<20}")
print("-" * 58)
for item in ["Mental Demand", "Effort", "Annoyance", "Success"]:
    a = nasa_output["items"][item]["A"]
    b = nasa_output["items"][item]["B"]
    print(f"{item:<18} {a['mean']:.1f} (SD={a['sd']:.1f}){'':<6} {b['mean']:.1f} (SD={b['sd']:.1f})")
print("-" * 58)
wa = nasa_output["workload_aggregate"]["A"]
wb = nasa_output["workload_aggregate"]["B"]
print(f"{'Workload Agg.':<18} {wa['mean']:.1f} (SD={wa['sd']:.1f}){'':<6} {wb['mean']:.1f} (SD={wb['sd']:.1f})")

print("\n── Satisfaction (1-7 scale) ─────────────────────────────────────────")
print(f"{'Measure':<20} {'System A':<20} {'System B':<20}")
print("-" * 60)
sa = satisfaction_output["A"]
sb = satisfaction_output["B"]
print(f"{'Overall':<20} {sa['mean']:.1f} (SD={sa['sd']:.1f}){'':<6} {sb['mean']:.1f} (SD={sb['sd']:.1f})")
aa = ai_satisfaction_output["A"]
ab = ai_satisfaction_output["B"]
print(f"{'AI Satisfaction':<20} {aa['mean']:.1f} (SD={aa['sd']:.1f}){'':<6} {ab['mean']:.1f} (SD={ab['sd']:.1f})")

print("\n── LD-Specific Items (System A only, 1-7 scale) ─────────────────────")
print(f"{'Item':<28} {'Mean':<8} {'SD':<8}")
print("-" * 44)
for label in sorted(ld_output.keys(), key=lambda x: -ld_output[x]["mean"]):
    s = ld_output[label]
    print(f"{label:<28} {s['mean']:.1f}{'':<4} {s['sd']:.1f}")

print("\n── Comparative Item (System A only, -2 to +2) ───────────────────────")
for pid in sorted(comp_values.keys(), key=lambda x: int(x[1:])):
    val = comp_values[pid]
    label = {-2: "Much Worse", -1: "Worse", 0: "Same", 1: "Better", 2: "Much Better"}.get(int(val), str(val))
    print(f"  {pid}: {val:+.0f} ({label})")
print(f"  Mean: {comparative_output['mean']:+.2f}")

print("\n" + "=" * 70)
print("Done.")
