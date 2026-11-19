"""
Syllabus Classification Functions (Hugging Face Inference Edition)

Author: Michael Kohlegger
"""

from difflib import SequenceMatcher, get_close_matches
from json import loads
import json
import os
import re
from openai import OpenAI
from requests import get

from .templates import Extraction
from .constant import EXTRACT_TEMPERATURE, EXTRACTION_PROMPT, INFERENCE_LENGTH


def get_framework(reference: str = "digicomp3-0") -> str:
    """Retrieves DigiComp Framework from github."""
    url = f"https://raw.githubusercontent.com/mckoh/{reference}/refs/heads/main/digital_compenteces.json"
    raw_json = get(url).content.decode("utf-8")
    raw_json = raw_json.replace("\n ", "").replace("\t ", "")
    raw_json = raw_json.replace("\n", "").replace("\t", "")
    raw_json = raw_json.strip()
    raw_json = " ".join(raw_json.split())
    return raw_json


def extract_valid_names(framework: str) -> list[str]:
    """Forms a list of valid competences from a dictionary."""
    nested_list = [[c["Competence Title"] for c in i["Competences"]] for i in loads(framework)]
    return [item for sublist in nested_list for item in sublist]


def is_valid_evidence(evidence: str, source_text: str, threshold: float = 0.80) -> bool:
    """Validates evidence strings against the source text."""
    norm_evidence = " ".join(evidence.lower().split())
    norm_source = " ".join(source_text.lower().split())

    if not norm_evidence:
        return False

    if norm_evidence in norm_source:
        return True

    ev_len = len(norm_evidence)
    src_len = len(norm_source)

    if ev_len > src_len:
        return SequenceMatcher(None, norm_evidence, norm_source).ratio() >= threshold

    window_size = min(src_len, int(ev_len * 1.15))
    step = max(1, ev_len // 4)

    for i in range(0, src_len - window_size + 1, step):
        window = norm_source[i : i + window_size]
        matcher = SequenceMatcher(None, norm_evidence, window)
        matched_chars = sum(block.size for block in matcher.get_matching_blocks())
        ratio = matched_chars / ev_len

        if ratio >= threshold:
            return True

    return False


def is_valid_competence(competence: str, valid_competences: list[str], threshold: float = 0.85) -> str | None:
    """Checks if an extracted competence is a valid competence label."""
    if not competence or not valid_competences:
        return None

    if competence in valid_competences:
        return competence

    cleaned_to_canonical = {
        re.sub(r"^\d+(\.\d+)*\s*", "", name).strip().lower(): name
        for name in valid_competences
    }
    cleaned_input = re.sub(r"^\d+(\.\d+)*\s*", "", competence).strip().lower()

    if cleaned_input in cleaned_to_canonical:
        return cleaned_to_canonical[cleaned_input]

    matches = get_close_matches(
        cleaned_input,
        list(cleaned_to_canonical.keys()),
        n=1,
        cutoff=threshold
    )
    if matches:
        return cleaned_to_canonical[matches[0]]

    return None


def _clean_json_response(raw_content: str) -> str:
    """Entfernt eventuelle Markdown-Codeblöcke um das JSON-Objekt."""
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract(
    text: str,
    api_key: str | None = None,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    base_url: str = "[https://router.huggingface.co/v1](https://router.huggingface.co/v1)",
    framework_ref: str = "digicomp3-0",
) -> Extraction:
    """Extracts and validates competences using Hugging Face Inference Providers."""
    resolved_key = api_key or os.getenv("HF_TOKEN")
    if not resolved_key:
        raise ValueError("Kein Hugging Face Token gefunden. Bitte HF_TOKEN setzen oder übergeben.")

    client = OpenAI(api_key=resolved_key, base_url=base_url)

    framework = get_framework(reference=framework_ref)
    valid_names = extract_valid_names(framework)
    framework_str = framework if isinstance(framework, str) else json.dumps(framework, ensure_ascii=False, indent=2)
    schema_str = json.dumps(Extraction.model_json_schema(), ensure_ascii=False)

    system_content = (
        f"{EXTRACTION_PROMPT}\n\n"
        f"KOMPETENZRAHMEN (Englisch):\n```json\n{framework_str}\n```\n\n"
        f"Antworte ausschließlich mit einem validen JSON-Objekt gemäß folgendem Schema:\n{schema_str}"
    )

    user_content = (
        f"KURSBESCHREIBUNG (Deutsch):\n```text\n{text}\n```"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=EXTRACT_TEMPERATURE,
        max_tokens=INFERENCE_LENGTH,
    )

    raw_json = _clean_json_response(resp.choices[0].message.content)
    result = Extraction.model_validate_json(raw_json)

    verified_competences = []
    seen_names = set()

    for c in result.competences:
        if not