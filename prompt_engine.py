"""
prompt_engine.py
Builds the structured prompt for the Gemini model and parses its JSON response.
"""

import json
import re

SYSTEM_INSTRUCTION = """You are an expert medical triage assistant AI.
Your role is to analyse the symptoms provided by the user and return a structured JSON response.

IMPORTANT RULES:
1. Always respond with ONLY valid JSON — no markdown fences, no extra text.
2. Base your triage on standard medical guidelines (similar to NHS 111 / CDC guidance).
3. Never claim to be a doctor. Always remind users that this is informational only.
4. Be concise and use plain language a non-medical person can understand.

Response JSON schema (strictly follow this):
{
  "triage_level": "<one of: SELF-CARE | SEE A DOCTOR | EMERGENCY>",
  "triage_color": "<one of: green | orange | red>",
  "triage_summary": "<1-2 sentence plain-language explanation of the triage decision>",
  "possible_conditions": [
    {
      "name": "<condition name>",
      "likelihood": "<Low | Moderate | High>",
      "brief_description": "<1 sentence description>"
    }
  ],
  "recommended_actions": ["<action 1>", "<action 2>", "..."],
  "warning_signs": ["<sign to watch for 1>", "..."],
  "recommended_doctors": [
    {
      "specialty": "<medical specialty, e.g. Dermatologist, Cardiologist, General Practitioner>",
      "why": "<1 sentence explaining why this specialist is relevant to the symptoms>",
      "example_doctors": [
        {
          "name": "<realistic full doctor name, e.g. Dr. Sarah Mitchell>",
          "qualification": "<e.g. MBBS, MD, FRCP>",
          "experience": "<e.g. 15 years in Dermatology>",
          "known_for": "<1 sentence about their expertise or notable area>"
        }
      ]
    }
  ],
  "disclaimer": "This is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment."
}

Triage level guidance:
- SELF-CARE (green): Minor symptoms manageable at home (e.g., mild cold, minor cuts, mild headache).
- SEE A DOCTOR (orange): Symptoms that need professional evaluation but are not immediately life-threatening.
- EMERGENCY (red): Life-threatening symptoms requiring immediate emergency services (call 911/999/112).
"""


def build_prompt(symptoms: str, age: str, gender: str, duration: str, extra_context: str) -> str:
    """Construct the user-turn prompt from form inputs."""
    parts = [f"Patient symptoms: {symptoms.strip()}"]
    if age:
        parts.append(f"Age: {age}")
    if gender and gender != "Prefer not to say":
        parts.append(f"Gender: {gender}")
    if duration:
        parts.append(f"Duration of symptoms: {duration}")
    if extra_context.strip():
        parts.append(f"Additional context: {extra_context.strip()}")

    parts.append(
        "\nPlease analyse these symptoms and return the structured JSON triage response exactly as specified."
    )
    return "\n".join(parts)


def parse_response(raw: str) -> dict:
    """
    Extract and parse the JSON object from the model's raw text response.
    Handles cases where the model wraps JSON in markdown fences.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Try to extract the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


TRIAGE_META = {
    "SELF-CARE": {
        "icon": "🟢",
        "color": "#1e7e34",
        "bg": "#d4edda",
        "border": "#c3e6cb",
        "label": "SELF-CARE",
    },
    "SEE A DOCTOR": {
        "icon": "🟠",
        "color": "#856404",
        "bg": "#fff3cd",
        "border": "#ffeeba",
        "label": "SEE A DOCTOR",
    },
    "EMERGENCY": {
        "icon": "🔴",
        "color": "#721c24",
        "bg": "#f8d7da",
        "border": "#f5c6cb",
        "label": "EMERGENCY",
    },
}


def get_triage_meta(level: str) -> dict:
    """Return display metadata for the given triage level."""
    return TRIAGE_META.get(level.upper(), TRIAGE_META["SEE A DOCTOR"])


LIKELIHOOD_ORDER = {"High": 0, "Moderate": 1, "Low": 2}


def sort_conditions(conditions: list) -> list:
    """Sort possible conditions by likelihood (High → Moderate → Low)."""
    return sorted(conditions, key=lambda c: LIKELIHOOD_ORDER.get(c.get("likelihood", "Low"), 3))
