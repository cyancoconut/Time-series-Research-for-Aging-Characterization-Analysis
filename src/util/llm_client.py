"""Provider-agnostic LLM client for cluster interpretation.

One small seam so :mod:`cluster.interpret_clusters` never imports a vendor SDK
directly: an :class:`LLMClient` protocol with ``interpret_cluster(signature)
-> ClusterLabel``, plus two implementations — :class:`OpenAILLMClient`
(current default backend) and :class:`AnthropicLLMClient`. The backend is
chosen via the battery-config key ``llm_provider`` (default ``"openai"``).

Credentials (``openai_api_key`` / ``anthropic_api_key``) are resolved like the
MinIO creds: battery config first, then the env var (``OPENAI_API_KEY`` /
``ANTHROPIC_API_KEY``), then the gitignored root ``config.json``.

The interpretation is advisory only — labels land in separate ``llm_*``
columns and never touch Capacity_py / SOH numerics.
"""

import json
import logging
import os
from typing import Protocol

from pydantic import BaseModel, Field

DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


class ClusterLabel(BaseModel):
    # Free-form: the LLM invents its own concise snake_case name for the
    # cluster (e.g. "cap_discharge_c2", "full_discharge_1c", "qocv_cha_c20").
    # Deliberately NOT constrained to the pipeline taxonomy, so the
    # interpretation is an independent second opinion next to `target`.
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMClient(Protocol):
    def interpret_cluster(self, signature: dict) -> ClusterLabel: ...


# Stable battery-domain context + taxonomy. This is the cached prefix: one API
# call per cluster, only the signature (user turn) varies, so the system block
# carries cache_control and must stay byte-identical across calls.
_SYSTEM_PROMPT = """\
You are a battery test-data analyst. A cycler runs periodic check-ups (CUs) on \
lithium-ion cells; the pipeline segments each CU into procedures and clusters \
the per-segment features with HDBSCAN. Your job: given the feature signature \
of ONE cluster (aggregated over its member segments), name what physical \
procedure that cluster is.

Feature semantics (all scale-free / chemistry-portable):
- Current_mean: signed mean current normalized by nominal capacity, i.e. a \
C-rate. Positive = charge, negative = discharge. abs_Current_mean = |.|.
- Voltage_max / Voltage_min / Voltage_range: segment voltage edges normalized \
by the cell's (V_max - V_min) window. Voltage_range ~ SoC swing of the segment.
- Duration_minutes: segment length. Duration_quartile = log1p(Duration_minutes).
- prev_end_voltage_norm: end-of-segment voltage of the nearest preceding \
non-pause segment, / V_max. ~1.0 means the segment started from a full cell.
- n_segments: cluster member count. majority_target: label the rule-based \
pipeline gave most members. bootstrap_label: weak rule-based name for \
leftover clusters.

Invent your OWN concise snake_case label that best describes the physical \
procedure the cluster is — do not limit yourself to any fixed taxonomy. Name \
what the numbers show, and put the characteristic quantities in the name when \
they matter, e.g. "cap_discharge_c2", "full_discharge_1c", "pulse_test", \
"pulse_restore", "qocv_charge_c20", "qocv_discharge_c20", "prep_charge", \
"cv_topoff_hold", "soc_adjust_partial_charge", "rest", "data_artifact". \
Typical procedures in a check-up are capacity tests, pulse tests (with \
restore pulses), quasi-OCV sweeps, preparation charges, SoC adjustments and \
rests — but name precisely what you see, not the nearest familiar thing: a 1C \
full discharge is "full_discharge_1c", not a C/2 capacity test. If the \
cluster mixes distinct behaviors (e.g. both current signs), say so in the \
name (prefix "mixed_") and explain in the rationale.

Judge from the numbers, not from majority_target / bootstrap_label — those \
report what a rule-based pipeline thinks, and second-guessing them is the \
point. Be honest in confidence: 0.9+ only for unambiguous signatures."""


class AnthropicLLMClient:
    """Claude backend via the official ``anthropic`` SDK.

    One ``messages.parse`` call per cluster with a Pydantic-validated
    structured output, adaptive thinking at medium effort, and prompt caching
    (``cache_control: ephemeral``) on the stable system prompt.
    """

    def __init__(self, cfg: dict):
        import anthropic  # lazy: only the anthropic backend needs the SDK

        api_key = (
            cfg.get("anthropic_api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
            or _root_config_key("anthropic_api_key")
        )
        # api_key=None lets the SDK fall back to its own env resolution
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = cfg.get("llm_model", DEFAULT_ANTHROPIC_MODEL)

    def interpret_cluster(self, signature: dict) -> ClusterLabel:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": "Cluster signature:\n"
                    + json.dumps(signature, indent=2, sort_keys=True, default=str),
                }
            ],
            output_format=ClusterLabel,
        )
        result = response.parsed_output
        if result is None:
            raise RuntimeError(
                f"Anthropic returned no parsable ClusterLabel "
                f"(stop_reason={response.stop_reason})"
            )
        return result


class OpenAILLMClient:
    """OpenAI(-compatible) backend via the official ``openai`` SDK.

    One ``chat.completions.parse`` call per cluster with a Pydantic-validated
    structured output. Chat Completions (rather than the Responses API) so
    OpenAI-compatible gateways work too — point ``llm_base_url`` at the
    server's OpenAI v1 root (e.g. ``https://chat.kiconnect.nrw/api/v1``);
    leave it unset for api.openai.com.
    """

    def __init__(self, cfg: dict):
        import openai  # lazy: only the openai backend needs the SDK

        api_key = (
            cfg.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY")
            or _root_config_key("openai_api_key")
        )
        # api_key=None lets the SDK fall back to its own env resolution
        self._client = openai.OpenAI(api_key=api_key, base_url=cfg.get("llm_base_url"))
        self._model = cfg.get("llm_model", DEFAULT_OPENAI_MODEL)

    def interpret_cluster(self, signature: dict) -> ClusterLabel:
        response = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Cluster signature:\n"
                    + json.dumps(signature, indent=2, sort_keys=True, default=str),
                },
            ],
            response_format=ClusterLabel,
        )
        result = response.choices[0].message.parsed
        if result is None:
            raise RuntimeError(
                "OpenAI returned no parsable ClusterLabel "
                f"(finish_reason={response.choices[0].finish_reason})"
            )
        return result


def make_llm_client(cfg: dict) -> LLMClient:
    provider = cfg.get("llm_provider", "openai")
    if provider == "openai":
        return OpenAILLMClient(cfg)
    if provider == "anthropic":
        return AnthropicLLMClient(cfg)
    raise ValueError(f"Unknown llm_provider {provider!r} (expected 'openai' or 'anthropic')")


def _root_config_key(key: str):
    """Read an optional credential from the gitignored root config.json."""
    for path in ("../config.json", "config.json"):
        try:
            with open(path) as f:
                value = json.load(f).get(key)
            if value:
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    logging.debug(f"{key} not found in config.json")
    return None
