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

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


class ClusterLabel(BaseModel):
    # Short snake_case name: "<procedure>[_<crate>]" (e.g. "full_discharge_1c",
    # "full_charge_c20", "pulse", "partial_cha_c3"). Deliberately NOT constrained to
    # the pipeline taxonomy, so the interpretation is an independent second
    # opinion next to `target`.
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
by the cell's (V_max - V_min) window. Voltage_range is the within-segment \
swing, but is distorted for charges/discharges because a PAU pause between \
procedures lets the cell voltage relax toward OCV before the segment starts. \
- true_voltage_range: the corrected SoC swing, ALREADY COMPUTED for you in the \
signature — use the supplied value, do NOT recompute it. (It is \
Voltage_max - prev_end_voltage_norm for charges and \
prev_end_voltage_norm - Voltage_min for discharges, given only so you can \
interpret it; it falls back to Voltage_range when prev_end_voltage_norm = -1, \
i.e. no predecessor.) This — not Voltage_range, not Voltage_max/Voltage_min — \
is the SOLE determinant of whether a segment is full or partial.
- Duration_minutes: segment length. Duration_quartile = log1p(Duration_minutes).
- prev_end_voltage_norm: end-of-segment voltage of the nearest preceding \
non-pause segment, normalized the same way as Voltage features: \
(V - V_min) / (V_max - V_min), so 0 = bottom rail, 1 = top rail. \
-1 means no predecessor exists (first segment of a cell). ~1.0 means the \
segment started from a full cell. \
Use prev_end_voltage_norm (not Voltage_min/Voltage_max) to judge the cell's \
SoC at the start of this segment — between procedures there is always a PAU \
pause where the cell relaxes toward OCV, so Voltage_min of a charge segment \
reflects the relaxed OCV, not the predecessor's end SoC.
- n_segments: cluster member count. majority_target: label the rule-based \
pipeline gave most members. bootstrap_label: weak rule-based name for \
leftover clusters.

Label format — SHORT, essentials only: "<procedure>" or "<procedure>_<crate>" \
in snake_case. Procedure is one of: "full_charge", "full_discharge", \
"partial_cha", "partial_dch", "pulse", "rest", "artifact", "unknown". \
Do NOT distinguish test pulses from restore pulses — both are just "pulse". \
There is NO "qocv" label: a full charge/discharge at any C-rate, including a \
very low-rate quasi-OCV-style sweep, is "full_charge"/"full_discharge" with its \
crate suffix (the pipeline decides quasi-OCV vs capacity downstream from the \
measured rate). \
If the signature is too ambiguous or self-contradictory to map to any \
procedure with reason, use the bare label "unknown" (no C-rate suffix) with a \
low confidence and explain what is unclear in the rationale — do NOT force a \
guess onto the nearest familiar procedure. \
Decide the label in this STRICT ORDER. STEP 1 — full vs partial, by \
true_voltage_range ALONE, hard cutoff 0.9: true_voltage_range >= 0.9 = FULL \
(spans essentially the whole SoC window); true_voltage_range < 0.9 = PARTIAL. \
Nothing else changes this — not duration, not C-rate. A long, low-rate sweep \
that only covers part of the window is PARTIAL. Voltage_max/Voltage_min \
touching a rail does NOT make a segment full — only true_voltage_range does. A \
positive-current charge that starts already full (prev_end_voltage_norm ~ 1.0) \
and ends at Voltage_max ~ 1.0 has true_voltage_range ~ 0: it is a PARTIAL \
top-up / CV hold, NEVER full_charge, however close Voltage_max is to 1.0. \
Symmetrically, a discharge starting already empty (prev_end_voltage_norm ~ 0) \
ending at Voltage_min ~ 0 has true_voltage_range ~ 0 and is PARTIAL, not \
full_discharge. STEP 2 — only now look at \
C-rate. If PARTIAL: the label is "partial_cha" (charge) or "partial_dch" \
(discharge), FULL STOP — a partial segment is NEVER "full_charge" or \
"full_discharge", however low its current or however long it lasts. If FULL: \
"full_charge" (charge) or "full_discharge" (discharge) by the sign of \
Current_mean, at WHATEVER C-rate — a very low-rate, full-window quasi-OCV-style \
sweep is still "full_discharge"/"full_charge", just with a low crate suffix; do \
NOT down-rank it to a partial. Append the C-rate only when it is meaningful \
and well-defined, written as "c2" (= C/2), "1c", "c20" etc., e.g. \
"full_discharge_c2", "full_discharge_1c", "full_charge_c20", "partial_cha_c3". \
If the cluster mixes distinct behaviors beyond that, prefix "mixed_" on the \
closest procedure. Nothing else goes in the label — no SoC windows, no \
delta-V, no qualifiers; put every detail in the rationale. Name what the \
numbers show, not the nearest familiar thing: a 1C full discharge is \
"full_discharge_1c", not a C/2 capacity test.

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
