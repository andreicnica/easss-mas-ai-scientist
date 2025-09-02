# cost_tracker.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json, os, time, threading

# ---- Configure prices here (USD per 1M tokens). Keep this editable. ----
# NOTE: Prices change. Treat these as defaults and override if needed.
DEFAULT_PRICING = {
    # chat/completions models (input, output)
    # "gpt-5-nano": {"input": 0.15, "output": 0.60},        # example numbers
    # "gpt-4o-mini": {"input": 0.15, "output": 0.60},       # example numbers
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # embeddings (input only)
    "text-embedding-3-small": {"input": 0.01},
    # "text-embedding-3-large": {"input": 0.13},
}
UNIT: float = 1000000.0


@dataclass
class UsageEvent:
    ts: float
    kind: str                    # "chat" | "embedding" | "other"
    model: str
    input_tokens: int
    output_tokens: int           # 0 for embeddings
    cost_usd: float
    meta: Dict[str, Any]

class CostTracker:
    """
    Append usage to a JSONL file and compute costs using a simple price table.
    Thread-safe appends; trivial and dependency-free.
    """
    def __init__(self, log_path: str = "usage_log.jsonl", pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self.log_path = log_path
        self.pricing = pricing or DEFAULT_PRICING
        self._lock = threading.Lock()

        # Ensure file exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                pass

    def _get_prices(self, model: str) -> Dict[str, float]:
        # Allow partial matches: if exact not found, try prefix stripping (e.g., "gpt-5-nano-1234")
        if model in self.pricing:
            return self.pricing[model]
        for key in self.pricing:
            if model.startswith(key):
                return self.pricing[key]
        # Fallback to zero (no surprise charges if unknown)
        return {"input": 0.0, "output": 0.0}

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int, kind: str) -> float:
        prices = self._get_prices(model)
        if kind == "embedding":
            # Embeddings are billed on input only
            return (input_tokens / UNIT) * float(prices.get("input", 0.0))
        # Chat/completions
        in_cost = (input_tokens / UNIT) * float(prices.get("input", 0.0))
        out_cost = (output_tokens / UNIT) * float(prices.get("output", prices.get("input", 0.0)))
        return in_cost + out_cost

    def _append(self, ev: UsageEvent) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev)) + "\n")

    def log(self, *, kind: str, model: str, input_tokens: int, output_tokens: int = 0, meta: Optional[Dict[str, Any]] = None) -> float:
        cost = self.compute_cost(model, input_tokens, output_tokens, kind)
        ev = UsageEvent(
            ts=time.time(),
            kind=kind,
            model=model,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cost_usd=float(cost),
            meta=meta or {},
        )
        self._append(ev)
        return cost

    # ----- Convenience: summarize the file -----
    def totals(self) -> Dict[str, Any]:
        total_cost = 0.0
        total_in = 0
        total_out = 0
        per_model = {}
        if not os.path.exists(self.log_path):
            return {"total_cost_usd": 0.0, "total_input_tokens": 0, "total_output_tokens": 0, "per_model": {}}

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total_cost += rec.get("cost_usd", 0.0)
                total_in += rec.get("input_tokens", 0)
                total_out += rec.get("output_tokens", 0)
                m = rec.get("model", "unknown")
                pm = per_model.setdefault(m, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0})
                pm["cost_usd"] += rec.get("cost_usd", 0.0)
                pm["input_tokens"] += rec.get("input_tokens", 0)
                pm["output_tokens"] += rec.get("output_tokens", 0)

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "per_model": per_model,
        }

def print_totals(log_path: str = "usage_log.jsonl"):
    print("\n ===== Usage/Cost summary =====")
    t = CostTracker(log_path=log_path)
    s = t.totals()
    print(f"Total input tokens: {s['total_input_tokens']:,}")
    print(f"Total output tokens: {s['total_output_tokens']:,}")
    if s["per_model"]:
        print("\nBy model:")
        for m, v in s["per_model"].items():
            print(f"  {m:<28}  ${v['cost_usd']:.6f}  in={v['input_tokens']:,}  out={v['output_tokens']:,}")\

    print(f"Total cost so far: ${s['total_cost_usd']:.6f}")
