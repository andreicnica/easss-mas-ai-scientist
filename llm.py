from openai import OpenAI
from cost_tracker import CostTracker
from typing import Optional
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


# =========================
# 2b) HuggingFace DeepSeek (local) wrapper
# =========================

class HFDeepseekLLM(BaseLLM):
    """
    Local HuggingFace model wrapper for DeepSeek chat models.
    - No API calls, runs fully local.
    - Uses chat template if provided by the tokenizer.
    - Logs token counts to CostTracker (assumed zero cost).
    """
    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-V2-Lite",  # choose your local model
        tracker: Optional[CostTracker] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        device_map: str = "auto",
        dtype: Optional[torch.dtype] = None,  # None => auto (bf16 if available)
        trust_remote_code: bool = True,
        max_new_tokens_default: int = 2048,
    ):
        self.model_id = model_id
        self.tracker = tracker
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens_default = int(max_new_tokens_default)

        if dtype is None:
            # Prefer bf16 if available; otherwise fallback to float16 or float32
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote_code
        )
        # Ensure pad token exists to avoid warnings during generate
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def _build_prompt(self, system: str, user: str) -> str:
        """
        Use chat template if available; otherwise fall back to a simple format.
        """
        messages = [
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user or ""},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback prompt (simple, generic)
        return (
            f"<|system|>\n{system}\n</|system|>\n"
            f"<|user|>\n{user}\n</|user|>\n"
            f"<|assistant|>\n"
        )

    @torch.inference_mode()
    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        prompt = self._build_prompt(system, user)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature is not None and self.temperature > 0.0
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens or self.max_new_tokens_default,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Separate the newly generated tokens from the prompt
        input_len = inputs["input_ids"].shape[1]
        gen_ids = gen_out[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # --- token accounting for your CostTracker (assumed zero-cost local run)
        in_toks = int(inputs["input_ids"].numel())
        out_toks = int(gen_ids.numel())
        if self.tracker is not None:
            meta = {
                "max_completion_tokens": max_tokens,
                "system_chars": len(system or ""),
                "user_chars": len(user or ""),
                "provider": "huggingface-local",
            }
            self.tracker.log(
                kind="chat",
                model=self.model_id,
                input_tokens=in_toks,
                output_tokens=out_toks,
                meta=meta,
            )

        return text



def get_model(tracker: CostTracker, model: str = "gpt-4.1-nano") -> BaseLLM:
    if "gpt" in model:
        return GPTLLM(model=model, tracker=tracker)
    elif "qwen" == model:
        llm = HFDeepseekLLM(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",  # <1B, chat-tuned
            tracker=tracker,
            temperature=0.2,
            top_p=0.95,
        )
        return llm
    else:
        raise ValueError(f"Unsupported model: {model}")
