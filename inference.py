
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ADAPTER_PATH = "adapter"

SYSTEM_PROMPT = """You are a strict tool-calling assistant. You have access to exactly 5 tools:
- weather: {"tool": "weather", "args": {"location": "string", "unit": "C|F"}}
- calendar: {"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
- convert: {"tool": "convert", "args": {"value": number, "from_unit": "string", "to_unit": "string"}}
- currency: {"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}
- sql: {"tool": "sql", "args": {"query": "string"}}

Rules:
1. If the user's request matches a tool, respond ONLY with <tool_call>JSON</tool_call>
2. If no tool fits, respond in plain text refusing politely.
3. Never emit a tool call for chitchat, jokes, or impossible tools."""

_tokenizer = None
_model = None

def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    _tokenizer.pad_token = _tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
    )
    _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    _model.eval()

def run(prompt: str, history: list[dict]) -> str:
    _load()

    # Build conversation context with history
    hist_text = ""
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        hist_text += f"{role}: {turn.get('content', '')}\n"

    full_input = f"{SYSTEM_PROMPT}\n\n{hist_text}User: {prompt}"
    formatted = f"<s>[INST] {full_input} [/INST]"

    inputs = _tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Only decode the NEW tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


if __name__ == "__main__":
    # Quick sanity checks
    print(run("What's the weather in London in Celsius?", []))
    print(run("Convert 50 km to miles", []))
    print(run("Tell me a joke", []))
    print(run("Now convert that to EUR", [
        {"role": "user", "content": "Convert 100 USD to GBP"},
        {"role": "assistant", "content": '<tool_call>{"tool": "currency", "args": {"amount": 100.0, "from": "USD", "to": "GBP"}}</tool_call>'}
    ]))
