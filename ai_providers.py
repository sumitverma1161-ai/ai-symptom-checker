"""
ai_providers.py
---------------
Dual-provider AI backend for the Symptom Checker app.

Supported providers:
  - Google Gemini  (model: gemini-3.8-flash)   — multimodal, supports image input
  - Groq           (model: llama3-70b-8192 or llama3-8b-8192) — text only, ultra-fast

Public API
----------
  call_triage(prompt, gemini_key, groq_key, provider, groq_model, image_bytes, image_mime)
      → dict   (parsed JSON result)

  call_lifestyle(topic, gemini_key, groq_key, provider, groq_model)
      → dict   (parsed JSON result)

Both functions support automatic fallback: if the active provider fails, the
other provider is tried automatically (if its key is configured).
"""

from __future__ import annotations

import json
import re
from typing import Literal

# ── Provider constants ────────────────────────────────────────────
PROVIDER_GEMINI = "Google Gemini"
PROVIDER_GROQ   = "Groq"

GEMINI_MODEL       = "gemini-3.8-flash"
GROQ_MODEL_LARGE   = "llama3-70b-8192"
GROQ_MODEL_SMALL   = "llama3-8b-8192"
GROQ_MODELS        = [GROQ_MODEL_LARGE, GROQ_MODEL_SMALL]

GROQ_API_BASE = "https://api.groq.com/openai/v1"


# ─────────────────────────────────────────────────────────────────
# Low-level provider calls
# ─────────────────────────────────────────────────────────────────

def _call_gemini_triage(
    prompt: str,
    api_key: str,
    system_instruction: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
) -> str:
    """Call Gemini and return raw response text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    if image_bytes:
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
            prompt,
        ]
    else:
        contents = prompt

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    return response.text


def _call_gemini_lifestyle(prompt: str, api_key: str, system_instruction: str) -> str:
    """Call Gemini for lifestyle guide and return raw response text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
            max_output_tokens=4096,
        ),
    )
    return response.text


def _call_groq(
    prompt: str,
    api_key: str,
    system_instruction: str,
    model: str = GROQ_MODEL_LARGE,
    max_tokens: int = 2048,
) -> str:
    """
    Call Groq's OpenAI-compatible chat endpoint and return the assistant message text.
    Note: Groq does not support image inputs — images are silently ignored.
    """
    import urllib.request

    url = f"{GROQ_API_BASE}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens":  max_tokens,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────
# JSON parse helper (shared)
# ─────────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
    """Strip markdown fences and parse the first JSON object."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


# ─────────────────────────────────────────────────────────────────
# Fallback helper
# ─────────────────────────────────────────────────────────────────

def _other_provider(provider: str) -> str:
    return PROVIDER_GROQ if provider == PROVIDER_GEMINI else PROVIDER_GEMINI


# ─────────────────────────────────────────────────────────────────
# Public: triage call
# ─────────────────────────────────────────────────────────────────

def call_triage(
    prompt: str,
    system_instruction: str,
    gemini_key: str,
    groq_key: str,
    provider: str,
    groq_model: str = GROQ_MODEL_LARGE,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
    auto_fallback: bool = True,
) -> tuple[dict, str]:
    """
    Run the triage prompt through the selected provider.

    Returns (result_dict, provider_used).
    Raises ProviderError if both providers fail.
    """
    primary   = provider
    secondary = _other_provider(provider)

    def _try(prov: str) -> str:
        if prov == PROVIDER_GEMINI:
            if not gemini_key:
                raise ValueError("Gemini API key is not configured.")
            return _call_gemini_triage(prompt, gemini_key, system_instruction, image_bytes, image_mime)
        else:
            if not groq_key:
                raise ValueError("Groq API key is not configured.")
            # Groq doesn't support images — strip note to avoid confusing the model
            text_only_prompt = prompt
            return _call_groq(text_only_prompt, groq_key, system_instruction, groq_model, max_tokens=2048)

    # ── Try primary ──────────────────────────────────────────────
    primary_error = None
    try:
        raw = _try(primary)
        return _parse(raw), primary
    except Exception as exc:
        primary_error = exc

    # ── Auto-fallback to secondary ───────────────────────────────
    if auto_fallback:
        secondary_key = groq_key if secondary == PROVIDER_GROQ else gemini_key
        if secondary_key:
            try:
                raw = _try(secondary)
                return _parse(raw), secondary
            except Exception as sec_exc:
                raise ProviderError(
                    primary=primary,
                    primary_error=primary_error,
                    secondary=secondary,
                    secondary_error=sec_exc,
                )

    raise ProviderError(primary=primary, primary_error=primary_error)


# ─────────────────────────────────────────────────────────────────
# Public: lifestyle guide call
# ─────────────────────────────────────────────────────────────────

def call_lifestyle(
    topic: str,
    system_instruction: str,
    gemini_key: str,
    groq_key: str,
    provider: str,
    groq_model: str = GROQ_MODEL_LARGE,
    auto_fallback: bool = True,
) -> tuple[dict, str]:
    """
    Run the lifestyle guide prompt through the selected provider.

    Returns (result_dict, provider_used).
    """
    prompt = f"Generate a comprehensive lifestyle guide on: {topic}"

    def _try(prov: str) -> str:
        if prov == PROVIDER_GEMINI:
            if not gemini_key:
                raise ValueError("Gemini API key is not configured.")
            return _call_gemini_lifestyle(prompt, gemini_key, system_instruction)
        else:
            if not groq_key:
                raise ValueError("Groq API key is not configured.")
            return _call_groq(prompt, groq_key, system_instruction, groq_model, max_tokens=4096)

    primary   = provider
    secondary = _other_provider(provider)
    primary_error = None

    try:
        raw = _try(primary)
        return _parse(raw), primary
    except Exception as exc:
        primary_error = exc

    if auto_fallback:
        secondary_key = groq_key if secondary == PROVIDER_GROQ else gemini_key
        if secondary_key:
            try:
                raw = _try(secondary)
                return _parse(raw), secondary
            except Exception as sec_exc:
                raise ProviderError(
                    primary=primary,
                    primary_error=primary_error,
                    secondary=secondary,
                    secondary_error=sec_exc,
                )

    raise ProviderError(primary=primary, primary_error=primary_error)


# ─────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────

class ProviderError(Exception):
    """Raised when all attempted providers fail."""

    def __init__(
        self,
        primary: str,
        primary_error: Exception,
        secondary: str | None = None,
        secondary_error: Exception | None = None,
    ):
        self.primary         = primary
        self.primary_error   = primary_error
        self.secondary       = secondary
        self.secondary_error = secondary_error

    def user_message(self) -> str:
        msg = f"❌ **{self.primary}** failed: `{self.primary_error}`"
        if self.secondary and self.secondary_error:
            msg += f"\n\n❌ Fallback to **{self.secondary}** also failed: `{self.secondary_error}`"
        return msg

    def __str__(self) -> str:
        return self.user_message()
