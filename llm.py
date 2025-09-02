from openai import OpenAI
from cost_tracker import CostTracker
from typing import Optional

# =========================
# 2) Tiny GPT wrapper
# =========================
class BaseLLM:
    def generate(self, system: str, user: str, max_tokens: int = 256) -> str:
        raise NotImplementedError

class GPTLLM(BaseLLM):
    _client: OpenAI

    def __init__(self, model: str = "gpt-4.1-nano", tracker: Optional[CostTracker] = None):
        self.model = model
        self.tracker = tracker  # <- NEW
        try:
            self._client = OpenAI()
        except Exception as e:
            raise RuntimeError("Install openai>=1.0 and set OPENAI_API_KEY") from e

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )

        # Robust usage extraction (field names may vary)
        usage = getattr(resp, "usage", None)
        in_toks = out_toks = 0
        if usage:
            in_toks = (
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "input_tokens", None)
                or 0
            ) or 0
            out_toks = (
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "output_tokens", None)
                or 0
            ) or 0

        if self.tracker is not None:
            meta = {
                "max_completion_tokens": max_tokens,
                "system_chars": len(system or ""),
                "user_chars": len(user or ""),
            }
            self.tracker.log(kind="chat", model=self.model, input_tokens=int(in_toks), output_tokens=int(out_toks), meta=meta)

        return resp.choices[0].message.content.strip()