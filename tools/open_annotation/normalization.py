"""Auditable text normalization, with no invented numeric/abbreviation reading."""
import re
import unicodedata

POLICY = "nfkc_punctuation_to_space_whitespace_collapse/v1"
APOSTROPHE_PROFILE = "nfkc_preserve_ascii_word_apostrophe_punctuation_space/v2"
# The 10-25/10-27 trees recorded only this free-text description, not the
# policy id.  Map that exact known text explicitly; do not fall back to the
# current POLICY for an unrecognised profile.
LEGACY_PROFILE_TEXT = "NFKC punctuation-to-space collapse-whitespace/v1; preserve raw; no invented spoken forms"
LEGACY_TEXT_PROFILES = {LEGACY_PROFILE_TEXT: POLICY}
SUPPORTED_POLICIES = (POLICY, APOSTROPHE_PROFILE)
# An uppercase English word is not evidence of an initialism. Keep this small,
# explicit vocabulary auditable rather than guessing from typography alone.
EXPLICIT_INITIALISMS = frozenset((
    "AI API ASR ASCII BBC BGM CCTV CD CNN CPU DDP DNA DVD EU FLAC GPT GPU HTML HTTP HTTPS "
    "ID IP LLM LM MV PC PDF PTN RNA SQL TV UK URL US USA USB VAD WIFI"
).split())


def _spelled_letter_runs(normalized):
    """Uppercase single-letter runs after the unchanged punctuation policy."""
    runs=[];run=[]
    for token in normalized.split()+[""]:
        if len(token)==1 and "A" <= token <= "Z":
            run.append(token)
        else:
            if len(run)>=2 and not (len(set(run))==1 and run[0] in {"A","I"}):runs.append(run)
            run=[]
    return runs


def normalize_with_audit(raw, policy=POLICY):
    if not isinstance(raw, str): raise TypeError("ASR text must be a string")
    if policy not in SUPPORTED_POLICIES: raise ValueError("unknown explicit normalization profile")
    nfkc = unicodedata.normalize("NFKC", raw)
    def preserved_apostrophe(i, c):
        return (policy == APOSTROPHE_PROFILE and c == "'" and 0 < i < len(nfkc)-1
                and unicodedata.category(nfkc[i-1]).startswith("L")
                and unicodedata.category(nfkc[i+1]).startswith("L"))
    punct = "".join(" " if unicodedata.category(c).startswith("P") and not preserved_apostrophe(i,c) else c
                    for i,c in enumerate(nfkc))
    normalized = " ".join(punct.split())
    reasons = []
    if any(unicodedata.category(c) == "Nd" for c in nfkc): reasons.append("numeric_spoken_form_unresolved")
    scripts = set()
    for c in nfkc:
        if not unicodedata.category(c).startswith("L"): continue
        name = unicodedata.name(c, "")
        script = "HAN" if name.startswith(("CJK ", "IDEOGRAPHIC ")) else name.split(" ", 1)[0]
        if script: scripts.add(script)
    if len(scripts) > 1: reasons.append("mixed_scripts_spoken_form_unresolved")
    uppercase_words=re.findall(r"(?<![A-Za-z])[A-Z]{2,}(?![A-Za-z])",nfkc)
    explicit_initialisms=sorted(set(uppercase_words) & EXPLICIT_INITIALISMS)
    letter_runs=_spelled_letter_runs(normalized)
    if explicit_initialisms:
        reasons.append("abbreviation_spoken_form_unresolved")
    if letter_runs:reasons.append("separated_uppercase_letters_spoken_form_unresolved")
    apostrophe_text = nfkc if policy == APOSTROPHE_PROFILE else raw
    if re.search(r"[^\W\d_]['’][^\W\d_]", apostrophe_text, re.UNICODE): reasons.append("word_internal_apostrophe_requires_review")
    if re.search(r"\d[.,]\d", nfkc): reasons.append("decimal_or_group_separator_requires_review")
    # This linear map is exact only when NFKC does not combine across raw
    # codepoint boundaries. Ambiguous cross-codepoint normalization is explicit.
    parts = []; spans = []
    for i, c in enumerate(raw):
        part = unicodedata.normalize("NFKC", c)
        parts.append(part); spans.extend([(i, i + 1)] * len(part))
    mapping = None; mapping_status = "requires_sequence_normalization_alignment"
    if "".join(parts) == nfkc:
        out = []; mapped = []; pending = []
        for c, span in zip(punct, spans):
            if c.isspace(): pending.append(span); continue
            if pending and out:
                out.append(" "); mapped.append((min(x[0] for x in pending), max(x[1] for x in pending)))
            pending.clear(); out.append(c); mapped.append(span)
        assert "".join(out) == normalized
        mapping = [{"normalized_index": i, "raw_start": a, "raw_end": b} for i, (a, b) in enumerate(mapped)]
        mapping_status = "exact_codepoint_half_open"
    else:
        reasons.append("normalization_sequence_mapping_unresolved")
    return {"policy": policy, "raw_text": raw, "nfkc_text": nfkc, "punctuation_space_text": punct,
            "normalized_text": normalized, "spoken_form_review_required": bool(reasons), "reasons": sorted(set(reasons)),
            "normalized_character_map": mapping, "mapping_status": mapping_status,
            "raw_offset_unit": "unicode_codepoint_half_open", "observed_letter_scripts": sorted(scripts),
            "spoken_form_policy": "explicit_initialisms_and_separated_uppercase_letters/v2",
            "observed_uppercase_words": uppercase_words,
            "explicit_initialisms_requiring_review": explicit_initialisms,
            "separated_uppercase_letter_runs": letter_runs}
