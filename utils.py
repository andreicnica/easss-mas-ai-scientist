# multi_agent_ai_scientist_peptides.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
from typing import Any, Iterable, List, Union, Optional
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embedding_model import EmbeddingCache
from datetime import datetime
from pathlib import Path
from typing import Any


# =========================
# 1) Hypothesis structure
# =========================
JSON_RULES_RE = re.compile(r"\{.*?\"rules\"\s*:\s*\[.*?\]\s*\}", re.DOTALL)

@dataclass
class Message:
    agent_name: str
    content: str

@dataclass(frozen=True)
class Hypothesis:
    rules: list[str]            # natural-language rule sentences
    msgs: list[Message]         # raw model outputs that produced/led to these rules

    def __add__(self, other: Hypothesis) -> Hypothesis:
        return Hypothesis(
            rules=self.rules + other.rules,
            msgs=self.msgs + other.msgs
        )


def canon_rules(rules: Iterable[str]) -> str:
    return " | ".join(sorted(rules))

def parse_hypothesis_old(text: str) -> Optional[list[str]]:
    """
    Accepts either a raw JSON object containing {"rules":[...]} or a string containing it.
    Returns list[str] (deduped, normalized) or None.
    """
    txt = text.strip()
    blob = txt if txt.startswith("{") else (JSON_RULES_RE.search(txt).group(0) if JSON_RULES_RE.search(txt) else None)
    if not blob:
        return None
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict) and isinstance(obj.get("rules"), list):
            rules = []
            for r in obj["rules"]:
                if isinstance(r, str) and r.strip():
                    rules.append(" ".join(r.split()).strip())  # canonicalize whitespace
            if rules:
                # dedupe while preserving order
                seen = set()
                out = []
                for r in rules:
                    if r not in seen:
                        out.append(r); seen.add(r)
                return out
    except Exception:
        return None
    return None


class RuleParseError(ValueError):
    pass

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _extract_json_block(s: str) -> Optional[str]:
    """If a JSON code block is present, return its contents; else None."""
    m = _JSON_FENCE_RE.search(s)
    return m.group(1) if m else None

def _coerce_to_obj(payload: Union[str, dict, list]) -> Any:
    """Accept raw dict/list, JSON string, or markdown-fenced JSON."""
    if isinstance(payload, (dict, list)):
        return payload
    if not isinstance(payload, str):
        raise RuleParseError(f"Unsupported payload type: {type(payload)}")

    # Try fenced JSON first
    fenced = _extract_json_block(payload)
    text = fenced if fenced is not None else payload.strip()

    # Strip leading labels like `[M:0] Basic:` if present
    # (grab substring starting from first `{` or `[` to end)
    first_bracket = min([i for i in [text.find("{"), text.find("[")] if i != -1], default=-1)
    if first_bracket > 0:
        text = text[first_bracket:]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuleParseError(f"Could not parse JSON: {e}") from e

def _pick_rules_container(obj: Any) -> Iterable[Any]:
    """Return the iterable that should contain rule strings."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Prefer common keys; fall back to the first list value
        for key in ("rules", "hypotheses", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        # Fallback: first list-valued field
        for v in obj.values():
            if isinstance(v, list):
                return v
        raise RuleParseError("Dict payload has no list field (e.g., 'rules').")
    raise RuleParseError(f"Unsupported top-level JSON type: {type(obj)}")

def _clean_rule(s: str) -> str:
    """Trim, collapse whitespace; keep original punctuation & motif syntax."""
    # Remove surrounding quotes/spaces and collapse internal whitespace
    s = " ".join(s.strip().split())
    return s

def parse_hypothesis(
    payload: Union[str, dict, list],
    *,
    min_len: int = 1,
    dedupe: bool = True
) -> List[str]:
    """
    Normalize various inputs into a clean list of rule strings.

    Accepts:
      - {"rules": [...]} (or 'hypotheses'/'items')
      - ["...", "..."]
      - JSON string for either of the above (with or without ```json fences)
      - Strings with leading labels like "[M:0] Basic:" before JSON
    """
    obj = _coerce_to_obj(payload)
    container = _pick_rules_container(obj)

    cleaned: List[str] = []
    seen = set()
    for item in container:
        if isinstance(item, str):
            r = _clean_rule(item)
        elif isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
            r = _clean_rule(item["text"])
        else:
            # Skip non-strings quietly; could also raise if you prefer strictness
            continue

        if len(r) < min_len:
            continue
        if dedupe:
            key = r.lower()
            if key in seen:
                continue
            seen.add(key)
        cleaned.append(r)

    if not cleaned:
        raise RuleParseError("No valid rule strings found after normalization.")
    return cleaned

def merge_hypothesis(hypothesis: list[Hypothesis]) -> Hypothesis:
    x = hypothesis[0] if hypothesis else Hypothesis([], [])
    for h in hypothesis[1:]:
        x += h
    return x


# --- NEW: utility to format transcripts compactly ---
def format_transcript(msgs: List[Message], max_turns: int = 12) -> str:
    snippet = msgs[-max_turns:]
    return "\n".join(f"{m.agent_name}: {m.content}" for m in snippet) if snippet else "(no prior messages)"


def log_transcript(transcript_label: str, msgs: list[Message]) -> None:
    print(f"--- >>> TRANSCRIPT::{transcript_label} >>> ---")
    for ix, m in enumerate(msgs):
        print(f"[M:{ix}] {m.agent_name}: {m.content}")
    print(f"--- <<< END::{transcript_label} <<< ---")


# =========================
# 5) Embedding utilities
# =========================
def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na*nb)

def cosine_dist(a: List[float], b: List[float]) -> float:
    return 1.0 - cosine_sim(a, b)

def diversity_stats_emb(hyps: List[str], emb: EmbeddingCache) -> Dict[str, float]:
    """Higher mean pairwise cosine distance -> more diverse."""
    if len(hyps) < 2:
        return {"num_unique": 1, "count": 1, "mean_pairwise_cosine_dist": 0.0}
    vecs = emb.embed_many(hyps)
    dists = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            dists.append(cosine_dist(vecs[i], vecs[j]))
    return {"num_unique": len(set(hyps)), "count": len(hyps), "mean_pairwise_cosine_dist": float(sum(dists)/len(dists))}

def plot_embedding(hyps: list[str], emb: EmbeddingCache, method: str = "pca", title: str = "Hypotheses map"):
    if not hyps:
        print("No hypotheses to plot."); return
    vecs = np.array(emb.embed_many(hyps), dtype=float)
    if method.lower() == "tsne":
        proj = TSNE(n_components=2, perplexity=min(30, max(5, len(hyps)//2)), init="random", random_state=0).fit_transform(vecs)
    else:
        proj = PCA(n_components=2, random_state=0).fit_transform(vecs)
    labels = [f"{h} rules" for h in hyps]
    plt.figure()
    plt.scatter(proj[:,0], proj[:,1])
    for (x,y), lab in zip(proj, labels):
        plt.annotate(lab, (x,y))
    plt.title(title); plt.xlabel("dim-1"); plt.ylabel("dim-2"); plt.tight_layout()
    plt.show()



# =========================
# 6) Greedy matching score vs GT
# =========================
def greedy_groundtruth_score(ground_truth_rules: List[str], proposed_rules: List[str], emb) -> Dict:
    """
    1) compute all pairwise distances GT x Proposed (via embeddings)
    2) repeatedly take the min-distance pair, remove both, until GT exhausted
    3) score = mean similarity over matched GT (1 - mean distance)
       (if proposed < GT, unmatched GT get sim=0)
    """
    gt = list(set(ground_truth_rules))
    prop = list(dict.fromkeys(proposed_rules))  # dedupe while preserving order

    # Pre-embed
    gt_vecs = emb.embed_many(gt) if gt else []
    pr_vecs = emb.embed_many(prop) if prop else []

    n_gt = len(gt)
    if n_gt == 0:
        return {"avg_distance": 0.0, "avg_similarity": 1.0, "pairs": []}

    # Build distance matrix
    D = np.ones((len(gt_vecs), max(1, len(pr_vecs))), dtype=float)
    for i, gv in enumerate(gt_vecs):
        for j, pv in enumerate(pr_vecs):
            D[i, j] = cosine_dist(gv, pv)

    used_gt = set()
    used_pr = set()
    pairs: List[Tuple[int,int,float]] = []  # (i_gt, j_pr, dist)

    while len(used_gt) < len(gt_vecs) and len(used_pr) < len(pr_vecs):
        best = None
        for i in range(len(gt_vecs)):
            if i in used_gt: continue
            for j in range(len(pr_vecs)):
                if j in used_pr: continue
                d = D[i, j]
                if best is None or d < best[2]:
                    best = (i, j, float(d))
        if best is None:
            break
        i, j, d = best
        used_gt.add(i); used_pr.add(j); pairs.append((i,j,d))

    distances = [d for _,_,d in pairs]
    distances += [1.0] * (n_gt - len(pairs))
    avg_dist = float(sum(distances) / max(1, n_gt))
    return {
        "avg_distance": avg_dist,
        "avg_similarity": 1.0 - avg_dist,
        "pairs": [
            {"gt": gt[i], "proposed": prop[j], "distance": d, "similarity": 1.0 - d}
            for (i,j,d) in pairs
        ],
        "unmatched_gt": [gt[i] for i in range(n_gt) if i not in used_gt]
    }

def score_hypothesis_set_vs_gt(hyps: List[str], emb: EmbeddingCache, ground_truth_rules: list[str]) -> Dict:
    return greedy_groundtruth_score(ground_truth_rules, hyps, emb)


def evaluate_merged(merged_rules: List[str], emb: EmbeddingCache, ground_truth_rules: list[str]):
    score = greedy_groundtruth_score(ground_truth_rules, merged_rules, emb)
    diversity = diversity_stats_emb(merged_rules, emb)
    return {"score": score, "diversity": diversity}

# =========================
# Logging helpers (JSONL)
# =========================

def append_jsonl(path: str | Path, data: dict) -> None:
    """Append a dict as one JSON line to the given file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # auto-add timestamp if not present
    if "timestamp" not in data:
        data = {"timestamp": datetime.now().isoformat(timespec="seconds"), **data}

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


@dataclass
class ArgumentsParent:
    model: str

    def __post_init__(self):
        if "gpt" in self.model:
            assert self.model in ["gpt-4.1-nano"], "GPT models are restricted to gpt-4.1-nano"