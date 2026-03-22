from __future__ import annotations

# System prompt used both during inference and as the LoRA training instruction.
# Keeping them identical ensures the fine-tuned model behaves the same as at inference.
SYSTEM_TEMPLATE = (
    "You are {business_name}'s assistant. "
    "Answer ONLY using the provided context. "
    "If the context does not contain enough information to answer, "
    'say "I don\'t have that information." '
    "Always cite the source IDs (e.g. [DOC-XXXX-0]) used in your answer. "
    "Do NOT use any knowledge outside the provided context."
)

USER_TEMPLATE = "Context:\n{context}\n\n---\n\nQuestion: {query}"

# ── Chat templates per model family ──────────────────────────────────────────
# llama_cpp's create_chat_completion handles these automatically when
# the correct chat_format is set, but we also expose them for training.

CHAT_FORMATS = {
    "phi3":    "phi3",
    "mistral": "mistral-instruct",
    "llama3":  "llama-3",
    "chatml":  "chatml",          # generic fallback used by many models
}


def build_system_prompt(business_name: str) -> str:
    return SYSTEM_TEMPLATE.format(business_name=business_name)


def build_user_prompt(context: str, query: str) -> str:
    return USER_TEMPLATE.format(context=context, query=query)


def build_chat_messages(
    business_name: str, context: str, query: str
) -> list[dict[str, str]]:
    """Standard OpenAI-style messages list. Used by llama_cpp chat completion."""
    return [
        {"role": "system", "content": build_system_prompt(business_name)},
        {"role": "user",   "content": build_user_prompt(context, query)},
    ]


def format_for_completion(business_name: str, context: str, query: str) -> str:
    """Fallback single-string prompt for models that don't support chat format."""
    system = build_system_prompt(business_name)
    user = build_user_prompt(context, query)
    return f"{system}\n\n{user}\n\nAnswer:"


def format_training_example(
    business_name: str, context: str, query: str, answer: str
) -> dict[str, str]:
    """Format a single training example for LoRA fine-tuning (ChatML style)."""
    return {
        "system": build_system_prompt(business_name),
        "user":   build_user_prompt(context, query),
        "assistant": answer,
    }
