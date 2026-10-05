"""
ai_providers.py
---------------
Dual-provider multimodal AI backend for the Symptom Checker app.

Supported providers & models
─────────────────────────────
  Google Gemini  →  gemini-3.8-flash            (text + vision)
  Groq           →  llama3-70b-8192             (text only)
                    llama3-8b-8192              (text only, faster)
                    meta-llama/llama-4-scout-17b-16e-instruct (vision — for image analysis)

Image encoding
──────────────
  Gemini  : raw bytes   → types.Part.from_bytes(data=..., mime_type=...)
  Groq    : base64 data URL  → "data:<mime>;base64,<b64string>"

Fallback logic
──────────────
  • Any exception from the primary provider triggers fallback (if enabled).
  • 503 / "high demand" / "overloaded" errors are detected and trigger
    fallback even when auto_fallback is False (always-fallback-on-503).

Public API
──────────
  call_triage(...)           → (dict, provider_used_str)
  call_lifestyle(...)        → (dict, provider_used_str)
  call_vision_analysis(...)  → (dict, provider_used_str)   ← NEW
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request

# ── Provider / model constants ────────────────────────────────────
PROVIDER_GEMINI = "Google Gemini"
PROVIDER_GROQ   = "Groq"

GEMINI_MODEL              = "gemini-3.8-flash"

GROQ_MODEL_LARGE          = "llama3-70b-8192"
GROQ_MODEL_SMALL          = "llama3-8b-8192"
GROQ_MODEL_VISION         = "meta-llama/llama-4-scout-17b-16e-instruct"   # Groq vision model

# Text-only models shown in the Settings dropdown
GROQ_MODELS               = [GROQ_MODEL_LARGE, GROQ_MODEL_SMALL]
# All Groq models (text + vision) — used internally
GROQ_ALL_MODELS           = [GROQ_MODEL_LARGE, GROQ_MODEL_SMALL, GROQ_MODEL_VISION]

GROQ_API_BASE = "https://api.groq.com/openai/v1"


# ─────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────

def _to_base64_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Encode raw image bytes as a base64 data URL for Groq vision."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ─────────────────────────────────────────────────────────────────
# 503 / overload detection
# ─────────────────────────────────────────────────────────────────

def _is_overload_error(exc: Exception) -> bool:
    """Return True when the exception looks like a 503 / capacity error."""
    msg = str(exc).lower()
    markers = ("503", "service unavailable", "overloaded", "high demand",
               "capacity", "rate limit", "too many requests", "429")
    return any(m in msg for m in markers)


# ─────────────────────────────────────────────────────────────────
# Low-level: Gemini calls
# ─────────────────────────────────────────────────────────────────

def _call_gemini(
    prompt: str,
    api_key: str,
    system_instruction: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> str:
    """
    Single unified Gemini call (text or text+image).
    Returns raw response text.
    """
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
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text


# ─────────────────────────────────────────────────────────────────
# Low-level: Groq calls  (OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────

def _build_groq_messages(
    prompt: str,
    system_instruction: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
) -> list[dict]:
    """
    Build the messages array for Groq.
    If image_bytes is provided, constructs a vision-compatible content list
    using base64 data URL format (required by llama-3.2-11b-vision-preview).
    """
    messages = [{"role": "system", "content": system_instruction}]

    if image_bytes:
        data_url = _to_base64_data_url(image_bytes, image_mime)
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})
    return messages


def _call_groq(
    prompt: str,
    api_key: str,
    system_instruction: str,
    model: str = GROQ_MODEL_LARGE,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
) -> str:
    """
    Call Groq's OpenAI-compatible endpoint.
    Automatically selects the vision model when image_bytes is provided
    (overrides the passed model to llama-3.2-11b-vision-preview).
    Returns raw assistant message text.
    """
    # Upgrade to vision model when an image is present
    if image_bytes and model not in (GROQ_MODEL_VISION,):
        model = GROQ_MODEL_VISION

    messages = _build_groq_messages(prompt, system_instruction, image_bytes, image_mime)

    url     = f"{GROQ_API_BASE}/chat/completions"
    payload = json.dumps({
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        # Wrap with status code in message so _is_overload_error() picks it up
        body = http_err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq HTTP {http_err.code}: {body}") from http_err

    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────
# JSON parse helper
# ─────────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
    """Strip markdown fences and parse the first JSON object."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _other_provider(provider: str) -> str:
    return PROVIDER_GROQ if provider == PROVIDER_GEMINI else PROVIDER_GEMINI


# ─────────────────────────────────────────────────────────────────
# Generic dispatcher  (used by all public call_* functions)
# ─────────────────────────────────────────────────────────────────

def _dispatch(
    prompt: str,
    system_instruction: str,
    gemini_key: str,
    groq_key: str,
    provider: str,
    groq_model: str,
    image_bytes: bytes | None,
    image_mime: str,
    max_tokens: int,
    temperature: float,
    auto_fallback: bool,
) -> tuple[str, str]:
    """
    Core dispatch: try primary provider, fall back to secondary on any error
    (always falls back on 503/overload regardless of auto_fallback flag).

    Returns (raw_text, provider_used).
    """
    primary   = provider
    secondary = _other_provider(provider)

    def _try(prov: str) -> str:
        if prov == PROVIDER_GEMINI:
            if not gemini_key:
                raise ValueError("Gemini API key is not configured.")
            return _call_gemini(
                prompt, gemini_key, system_instruction,
                image_bytes=image_bytes, image_mime=image_mime,
                temperature=temperature, max_output_tokens=max_tokens,
            )
        else:
            if not groq_key:
                raise ValueError("Groq API key is not configured.")
            return _call_groq(
                prompt, groq_key, system_instruction,
                model=groq_model, max_tokens=max_tokens,
                temperature=temperature,
                image_bytes=image_bytes, image_mime=image_mime,
            )

    # ── Try primary ──────────────────────────────────────────────
    primary_error: Exception | None = None
    try:
        return _try(primary), primary
    except Exception as exc:
        primary_error = exc

    # ── Decide whether to fall back ──────────────────────────────
    # Always fall back on 503/overload; otherwise respect the toggle.
    should_fallback = auto_fallback or _is_overload_error(primary_error)

    if should_fallback:
        secondary_key = groq_key if secondary == PROVIDER_GROQ else gemini_key
        if secondary_key:
            try:
                return _try(secondary), secondary
            except Exception as sec_exc:
                raise ProviderError(
                    primary=primary,
                    primary_error=primary_error,
                    secondary=secondary,
                    secondary_error=sec_exc,
                )

    raise ProviderError(primary=primary, primary_error=primary_error)


# ─────────────────────────────────────────────────────────────────
# Public: triage  (symptom checker)
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
    """Returns (parsed_result_dict, provider_used)."""
    raw, used = _dispatch(
        prompt=prompt,
        system_instruction=system_instruction,
        gemini_key=gemini_key,
        groq_key=groq_key,
        provider=provider,
        groq_model=groq_model,
        image_bytes=image_bytes,
        image_mime=image_mime,
        max_tokens=2048,
        temperature=0.2,
        auto_fallback=auto_fallback,
    )
    return _parse(raw), used


# ─────────────────────────────────────────────────────────────────
# Public: lifestyle guide
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
    """Returns (parsed_guide_dict, provider_used)."""
    prompt = f"Generate a comprehensive lifestyle guide on: {topic}"
    raw, used = _dispatch(
        prompt=prompt,
        system_instruction=system_instruction,
        gemini_key=gemini_key,
        groq_key=groq_key,
        provider=provider,
        groq_model=groq_model,
        image_bytes=None,
        image_mime="image/jpeg",
        max_tokens=4096,
        temperature=0.4,
        auto_fallback=auto_fallback,
    )
    return _parse(raw), used


# ─────────────────────────────────────────────────────────────────
# Public: vision analysis  (email text / email screenshot)
# ─────────────────────────────────────────────────────────────────

VISION_SYSTEM_INSTRUCTION = """You are an expert medical and health information analyst.
Your task is to analyse the provided content — which may be email text, an email screenshot, or both — and extract any health-related information.

IMPORTANT RULES:
1. Always respond with ONLY valid JSON — no markdown fences, no extra text.
2. Be concise and use plain language.
3. If an image is provided, carefully read all visible text and visual elements in the image.
4. If both text and image are provided, combine insights from both.
5. Never claim to be a doctor. This is informational only.

Response JSON schema (strictly follow this):
{
  "summary": "<2-3 sentence overview of what the email/content is about>",
  "health_topics_detected": ["<topic 1>", "<topic 2>"],
  "key_medical_terms": ["<term 1>", "<term 2>"],
  "symptoms_mentioned": ["<symptom 1>", "<symptom 2>"],
  "medications_mentioned": ["<medication 1>", "<medication 2>"],
  "urgency_level": "<one of: LOW | MEDIUM | HIGH | EMERGENCY>",
  "urgency_reason": "<1 sentence explaining the urgency level>",
  "recommended_actions": ["<action 1>", "<action 2>"],
  "red_flags": ["<concerning phrase or finding 1>", "<concerning phrase or finding 2>"],
  "disclaimer": "This analysis is for informational purposes only and does not constitute medical advice."
}
"""


def call_vision_analysis(
    email_text: str,
    gemini_key: str,
    groq_key: str,
    provider: str,
    groq_model: str = GROQ_MODEL_LARGE,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
    auto_fallback: bool = True,
) -> tuple[dict, str]:
    """
    Analyse email text and/or an email screenshot image.

    Image encoding:
      Gemini → raw bytes via types.Part.from_bytes()
      Groq   → base64 data URL via _to_base64_data_url() — uses llama-3.2-11b-vision-preview

    Returns (parsed_analysis_dict, provider_used).
    """
    parts = []
    if email_text.strip():
        parts.append(f"Email content to analyse:\n\n{email_text.strip()}")
    if image_bytes:
        parts.append("An email screenshot has also been provided. Please read all text and visual elements in the image carefully.")
    if not parts:
        raise ValueError("Provide at least email text or an image to analyse.")

    prompt = "\n\n".join(parts) + "\n\nPlease return the structured JSON analysis exactly as specified."

    raw, used = _dispatch(
        prompt=prompt,
        system_instruction=VISION_SYSTEM_INSTRUCTION,
        gemini_key=gemini_key,
        groq_key=groq_key,
        provider=provider,
        groq_model=groq_model,
        image_bytes=image_bytes,
        image_mime=image_mime,
        max_tokens=2048,
        temperature=0.2,
        auto_fallback=auto_fallback,
    )
    return _parse(raw), used


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
        elif self.secondary:
            msg += f"\n\n_(Auto-fallback to **{self.secondary}** was attempted but its key is not configured.)_"
        return msg

    def __str__(self) -> str:
        return self.user_message()
