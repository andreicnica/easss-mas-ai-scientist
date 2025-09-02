"""
Request key
export OPENAI_API_KEY=<your-openai-api-key>
or
os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>
"""
# multi_agent_ai_scientist_peptides.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Any, Literal
from typing import List, Tuple, Optional

from simple_parsing import ArgumentParser

from cost_tracker import CostTracker, print_totals
from embedding_model import get_embedding_model
from llm import BaseLLM, GPTLLM, get_model
from problem_description import PROBLEM_DESCRIPTION, GROUND_TRUTH_RULES
from utils import Message, Hypothesis, parse_hypothesis, format_transcript, ArgumentsParent, append_jsonl
from utils import merge_hypothesis, log_transcript, evaluate_merged


# %%
# =========================
# 1) Agent
# =========================

@dataclass
class Agent:
    name: str
    llm: BaseLLM
    system_prompt: str
    user_prompt: str

    def system(self) -> str:
        return self.system_prompt

    def user(self) -> str:
        return self.user_prompt

    def propose(self) -> Optional[Hypothesis]:
        out = self.llm.generate(self.system(), self.user())
        h_list = parse_hypothesis(out)
        return Hypothesis(h_list if h_list else list(), [Message(self.name, out)])

    # --- NEW: context-aware reply for multi-turn orchestration ---
    def reply(self, transcript: List[Message], prompt: str, expect_rules: bool = False, max_tokens: int = 1024):
        user = USER_PROMPT.format(
            PROBLEM=PROBLEM_DESCRIPTION,
            PROMPT=(
                f"{prompt}\n\n"
                "Conversation so far (most recent last):\n" +
                format_transcript(transcript)
            )
        )
        out = self.llm.generate(self.system(), user, max_tokens=max_tokens)
        msg = Message(self.name, out)
        if expect_rules:
            rules = parse_hypothesis(out) or []
            return msg, Hypothesis(rules, [msg])
        return msg, None


# %%
# =========================
# 2) Orchestration registry + runner (standardized)
# =========================

@dataclass
class OrchestrationResult:
    name: str
    hypothesis: Hypothesis
    transcript: List[Message]
    extra: Dict[str, Any]

@dataclass
class CombinedOutcome:
    merged: Hypothesis                # merged rules across orchestrations
    per_run: List[OrchestrationResult]  # each orchestration's standardized result

def _normalize_result(name: str, result: Any) -> OrchestrationResult:
    """
    Accepts:
      - (Hypothesis, List[Message])
      - (Hypothesis, List[Message], extra)  # extra can be dict or any payload
      - OrchestrationResult
    Returns OrchestrationResult.
    """
    if isinstance(result, OrchestrationResult):
        res = result
    else:
        hyp, transcript, *rest = result
        payload = rest[0] if rest else None
        if payload is None:
            extra = {}
        elif isinstance(payload, dict):
            extra = payload
        else:
            extra = {"payload": payload}  # e.g., verdict list from propose_verify
        res = OrchestrationResult(name=name, hypothesis=hyp, transcript=transcript, extra=extra)

    # side-effect: log transcript uniformly
    log_transcript(res.name, res.transcript)
    return res

def run_orchestrations(orchestrations: Dict[str, Callable[[], Any]]) -> CombinedOutcome:
    """
    Run each orchestration (zero-arg callable), standardize outputs, log transcripts,
    and return merged hypothesis + per-run results.
    """
    results: List[OrchestrationResult] = []
    for name, fn in orchestrations.items():
        out = fn()
        results.append(_normalize_result(name, out))

    merged_h = merge_hypothesis([r.hypothesis for r in results])
    return CombinedOutcome(merged=merged_h, per_run=results)

# %%
# =========================
# 3) Orchestrations
# =========================

SYSTEM_PROMPT = """
Propose concise rule statements that, together, explain why 
some peptides are functional. Use the required JSON format with key 'rules'. 
Rules should be testable, compositional.

{ROLE}
"""

USER_PROMPT = """
Problem is:
{PROBLEM}

{PROMPT}
"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# --- Simple agent generations
def collect_proposals(agents: List[Agent], rounds: int = 1) -> Tuple[Hypothesis, List[Message]]:
    transcript: List[Message] = []
    collected: List[str] = []
    for a in agents:
        for _ in range(rounds):
            hyp = a.propose()
            transcript.extend(hyp.msgs)
            if hyp and hyp.rules:
                collected.extend(hyp.rules)

    return Hypothesis(collected, transcript), transcript


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# --- NEW: Debate orchestration ---
DEBATE_INSTRUCTION = (
    "Debate to discover the best minimal rule set. In your turn: "
    "(1) briefly critique the latest idea(s); (2) propose at most ONE new or refined rule; "
    "end your message with the JSON object {'rules': [...]} (can be empty if you only critique)."
)

MODERATOR_SYNTHESIS_INSTRUCTION = (
    "Synthesize the debate into the best 3–5 rules. Avoid duplicates, keep rules testable, "
    "and output ONLY the JSON object {'rules': [...]}."
)

def debate_hypothesis(agents: List[Agent], rounds: int = 2) -> Tuple[Hypothesis, List[Message]]:
    assert len(agents) >= 2, "Need at least two agents for debate."
    transcript: List[Message] = []
    collected: List[str] = []

    for _ in range(rounds):
        for a in agents:
            msg, hyp = a.reply(transcript, DEBATE_INSTRUCTION, expect_rules=True)
            transcript.append(msg)
            if hyp and hyp.rules:
                collected.extend(hyp.rules)

    # Moderator: reuse first agent's LLM to keep it simple
    moderator = Agent(
        name="Moderator",
        llm=agents[0].llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You are a neutral moderator synthesizing consensus rules."),
        user_prompt=""  # unused for reply()
    )
    mod_msg, final_h = moderator.reply(
        transcript,
        MODERATOR_SYNTHESIS_INSTRUCTION,
        expect_rules=True,
    )
    transcript.append(mod_msg)

    if final_h and final_h.rules:
        return final_h, transcript

    # Fallback: unify collected rules if moderator failed
    # Dedupe preserving order
    seen = set()
    fallback = [r for r in collected if not (r in seen or seen.add(r))]
    return Hypothesis(fallback, transcript), transcript

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# --- NEW: Propose→Verify orchestration ---
PROPOSE_PROMPT = (
    "Propose at most 3 concise rules that explain functionality. "
    "End ONLY with the JSON object {'rules': [...]}."
)

VERIFIER_PROMPT_TMPL = (
    "Evaluate ONLY the given rules for plausibility on the dataset. "
    "For EACH rule, output a verdict with status one of OK / INCORRECT / UNCERTAIN and a one-sentence reason. "
    "Do NOT propose new rules.\n\n"
    "Rules to evaluate (JSON): {RULES}\n\n"
    "Return ONLY the JSON object:\n"
    "{{\"verdicts\": [{{\"rule\": str, \"status\": \"OK|INCORRECT|UNCERTAIN\", \"reason\": str}}, ...]}}"
)

REVISE_PROMPT_TMPL = (
    ">Update your rule set using the verifier's feedback. "
    "Keep OK rules, revise or drop INCORRECT, and consider clarifying UNCERTAIN. "
    "Return ONLY the JSON object {{'rules': [...]}}.\n\n"
    "Verifier feedback: {VERDICTS}\n"
    "Your previous rules: {RULES}"
)

JSON_VERDICTS_RE = re.compile(r"\{.*?\"verdicts\"\s*:\s*\[.*?\]\s*\}", re.DOTALL)

def parse_verdicts(text: str) -> Optional[list[dict]]:
    """
    Parse {"verdicts":[{"rule":..., "status":"OK/INCORRECT/UNCERTAIN", "reason":"..."}]}
    """
    txt = text.strip()
    blob = txt if txt.startswith("{") else (JSON_VERDICTS_RE.search(txt).group(0) if JSON_VERDICTS_RE.search(txt) else None)
    if not blob:
        return None
    try:
        obj = json.loads(blob)
        v = obj.get("verdicts")
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if not isinstance(item, dict): continue
                rule = item.get("rule")
                status = item.get("status")
                reason = item.get("reason", "")
                if isinstance(rule, str) and isinstance(status, str):
                    cleaned.append({"rule": " ".join(rule.split()),
                                    "status": status.strip().upper(),
                                    "reason": reason})
            return cleaned or None
    except Exception:
        return None
    return None


def propose_verify(expert: Agent, verifier: Agent, iterations: int = 3) -> Tuple[Hypothesis, List[Message], Dict[str, Any]]:
    transcript: List[Message] = []
    current_rules: List[str] = []
    last_verdicts: Optional[list[dict]] = None

    for _ in range(iterations):
        # Expert proposes
        msg1, h1 = expert.reply(transcript, PROPOSE_PROMPT, expect_rules=True)
        transcript.append(msg1)
        if not h1 or not h1.rules:
            # nothing new; continue loop
            continue
        current_rules = h1.rules

        # Verifier judges (no new rules)
        v_prompt = VERIFIER_PROMPT_TMPL.format(RULES=json.dumps(current_rules, ensure_ascii=False))
        vmsg, _ = verifier.reply(transcript, v_prompt, expect_rules=False)
        transcript.append(vmsg)
        last_verdicts = parse_verdicts(vmsg.content) or []

        # Expert revises based on feedback
        r_prompt = REVISE_PROMPT_TMPL.format(
            VERDICTS=json.dumps(last_verdicts, ensure_ascii=False),
            RULES=json.dumps(current_rules, ensure_ascii=False)
        )
        msg2, h2 = expert.reply(transcript, r_prompt, expect_rules=True)
        transcript.append(msg2)
        if h2 and h2.rules:
            current_rules = h2.rules

    final_h = Hypothesis(current_rules, transcript)
    return final_h, transcript, {"verdicts": last_verdicts}


# %%
# =========================
# 7) Demo
# =========================


@dataclass
class Arguments(ArgumentsParent):
    experiment_name: str = "default"
    # experiments: tuple[str] = ("BasicProposal", "CollectProposals", "Debate", "ProposeVerify")
    experiments: tuple[str] = ("BasicProposal",)
    rounds: int = 2
    model: Literal["gpt-4.1-nano", "qwen"] = "gpt-4.1-nano"


def basic_agent(llm: GPTLLM, rounds: int) -> Callable[[], Any]:
    agent = Agent(
        name="Basic",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE=""),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="")
    )
    return partial(collect_proposals, [agent], rounds=rounds)


def default_propose(llm: GPTLLM, rounds: int) -> Callable[[], Any]:
    creative = Agent(
        name="Creative",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You are a very creative researcher who explores bold ideas."),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="Think outside the box and propose diverse patterns.")
    )
    skeptical = Agent(
        name="Skeptical",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You are a skeptical scientist who stresses falsifiability and minimality."),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="Focus on small, testable rules and question assumptions.")
    )
    return partial(collect_proposals, [creative, skeptical], rounds=rounds)


def debate_experiment(llm: GPTLLM, rounds: int) -> Callable[[], Any]:
    creative = Agent(
        name="Creative",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You are a very creative researcher who explores bold ideas."),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="Think outside the box and propose diverse patterns.")
    )
    skeptical = Agent(
        name="Skeptical",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You are a skeptical scientist who stresses falsifiability and minimality."),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="Focus on small, testable rules and question assumptions.")
    )
    return partial(debate_hypothesis, [creative, skeptical], rounds=rounds)


def expert_verifier(llm: GPTLLM, rounds: int) -> Callable[[], Any]:
    orchestrations = {}
    expert = Agent(
        name="Expert",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You are an expert in protein engineering; concise and precise."),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="Propose just 1–3 rules.")
    )
    verifier_agent = Agent(
        name="Verifier",
        llm=llm,
        system_prompt=SYSTEM_PROMPT.format(ROLE="You only evaluate the expert's rules; do not propose new rules."),
        user_prompt=USER_PROMPT.format(PROBLEM=PROBLEM_DESCRIPTION, PROMPT="Judge rules only.")
    )
    return partial(propose_verify, expert, verifier_agent, iterations=rounds)


def demo(args: Arguments) -> None:
    # Embedding model with cache

    usage_file = "usage_log.jsonl"
    tracker = CostTracker(log_path=usage_file)  # optionally pass custom pricing=dict
    emb = get_embedding_model(tracker, cache_path="embeddings_cache.json")
    llm = get_model(tracker, args.model)

    orchestrations: dict[str, Callable[[], Any]] = {}

    # -- simple agents
    orchestrations["BasicProposal"] = basic_agent(llm, rounds=args.rounds)
    orchestrations["CollectProposals"] = default_propose(llm, rounds=args.rounds)
    orchestrations["Debate"] = debate_experiment(llm, rounds=args.rounds)
    orchestrations["ProposeVerify"] = expert_verifier(llm, rounds=args.rounds)

    orchestrations = {k: v for k, v in orchestrations.items() if k in args.experiments}

    # --- Run all, standardize, merge ---
    combined = run_orchestrations(orchestrations)

    # Print per-orchestration rules
    for res in combined.per_run:
        print(f"\n=== {res.name} Final Rules ===")
        for r in res.hypothesis.rules:
            print(f"  • {r}")

    # Combined rules (deduped & merged)
    merged_rules = list(combined.merged.rules)
    print("\n=== Combined (Merged) Rules ===")
    for r in merged_rules:
        print(f"  • {r}")

    # Score vs GT + diversity
    metrics = evaluate_merged(merged_rules, emb, GROUND_TRUTH_RULES)
    s = metrics["score"]
    div = metrics["diversity"]

    print("\n=== Combined Set score vs Ground Truth (greedy) ===")
    if s.get("pairs"):
        print("Matched pairs:")
        for p in s["pairs"]:
            print(f"  - SIM={p['similarity']:.2f} :: GT='{p['gt']}'  <->  PROPOSED='{p['proposed']}'")
    if s.get("unmatched_gt"):
        print("Unmatched GT:", s["unmatched_gt"])
    print(f"Average distance    (lower is better): {s['avg_distance']:.3f}")
    print(f"Average similarity (higher is better): {s['avg_similarity']:.3f}")

    print("\n=== Diversity (embedding) - higher is better ===")
    for k, v in div.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    # Later, in any script:
    print_totals(usage_file)

    # # Example of accessing orchestration-specific extras
    # pv = next((r for r in combined.per_run if r.name == "ProposeVerify"), None)
    # if pv and pv.extra.get("verdicts") is not None:
    #     print("\n=== ProposeVerify verdicts (latest) ===")
    #     for v in pv.extra["verdicts"]:
    #         print(f"- [{v['status']}] {v['rule']} :: {v.get('reason','')}")

    # Build a dict with only what you want
    save_info = {
        "experiment_name": args.experiment_name,
        "experiments": args.experiments,
        "avg_similarity": metrics["score"]["avg_similarity"],
        "diversity": metrics["diversity"],
        "rounds": args.rounds,
        "model": args.model,
        "merged_rules": merged_rules,
    }

    append_jsonl("experiment_log.jsonl", save_info)
    print("[log] appended to experiment_log.jsonl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Arguments, dest="args")
    args = parser.parse_args().args
    demo(args)
