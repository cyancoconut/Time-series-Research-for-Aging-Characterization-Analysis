"""Helpers for the ``procedure_filter`` config key.

``procedure_filter`` names the check-up programme(s) whose Prozedur / test
files should flow through the pipeline. It accepts either a single substring
(str) or a list of substrings; a value matches when **any** entry is a
substring of the text. ``None`` means "no filter" (allow everything).
"""


def as_filter_list(procedure_filter):
    """Normalize ``procedure_filter`` to a list of substrings, or ``None``.

    - ``None`` -> ``None`` (no filter, allow all)
    - ``"jri_CU"`` -> ``["jri_CU"]``
    - ``["jri_CU", "jri_Char"]`` -> the list unchanged (falsy entries dropped)
    """
    if procedure_filter is None:
        return None
    if isinstance(procedure_filter, str):
        return [procedure_filter]
    return [f for f in procedure_filter if f]


def matches_any(text, procedure_filter):
    """True if ``text`` contains any of the filter substrings.

    ``procedure_filter`` may be a str, a list of str, or ``None`` (matches all).
    """
    filters = as_filter_list(procedure_filter)
    if filters is None:
        return True
    return any(f in text for f in filters)
