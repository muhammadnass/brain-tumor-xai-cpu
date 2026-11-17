"""
evaluation/case_ref.py
======================
Derives a study case reference from an uploaded NIfTI filename (option C).

Rules
-----
1. Strip .nii / .nii.gz / .dcm extensions.
2. Strip common modality suffixes (case-insensitive):
       _T1, _T1c, _T2, _FLAIR, _t1, _t1c, _t2, _flair,
       _T1w, _T1ce, _T1gd, T1_, T2_, FLAIR_
3. Strip trailing underscores, hyphens, dots.
4. Clip to 64 characters.
5. Fall back to "CASE_<timestamp>" if the result is empty.

The derived ref is presented to the evaluator as an editable pre-filled
field — they can correct it before saving if the filename had PHI embedded.
The system never writes the raw original filename to the database.

Example
-------
    "PT001_T1.nii.gz"   → "PT001"
    "BRATS_001_T1c.nii" → "BRATS_001"
    "sub-07_T1w.nii"    → "sub-07"
    "patient_scan.nii"  → "patient_scan"   (no modality suffix matched)
"""

import re
from datetime import datetime, timezone

# Ordered longest-first so greedy matching hits the right suffix
_MODALITY_SUFFIXES = [
    "_T1ce", "_T1gd", "_T1c", "_T1w", "_T1",
    "_T2w", "_T2",
    "_FLAIR",
    "_t1ce", "_t1gd", "_t1c", "_t1w", "_t1",
    "_t2w", "_t2",
    "_flair",
    "T1_", "T2_", "FLAIR_",
]

_EXT_RE = re.compile(r'(\.nii\.gz|\.nii|\.dcm)$', re.IGNORECASE)
_TRAIL_RE = re.compile(r'[_\-\.]+$')


def derive_case_ref(filename: str) -> str:
    """
    Derive a PHI-free case reference from a NIfTI filename.

    Args:
        filename: Original filename (basename only, not full path).

    Returns:
        A short alphanumeric-plus-hyphen/underscore case reference string,
        max 64 characters, safe to store as the study identifier.
    """
    # 1. Strip path components the caller may have included
    name = filename.strip().split('/')[-1].split('\\')[-1]

    # 2. Strip extension(s)
    name = _EXT_RE.sub('', name)

    # 3. Strip modality suffix (first match wins, longest-first ordering)
    for suffix in _MODALITY_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
        # Also handle upper-case prefix variants (T1_case001 → case001)
        if name.startswith(suffix.lstrip('_')):
            name = name[len(suffix.lstrip('_')):]
            break

    # 4. Remove trailing punctuation
    name = _TRAIL_RE.sub('', name)

    # 5. Clip
    name = name[:64].strip()

    # 6. Fallback
    if not name:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        name = f"CASE_{ts}"

    return name


def suggest_case_ref(t1_filename: str, existing_refs: list[str] | None = None) -> str:
    """
    Derive a case ref and ensure it doesn't collide with already-used refs
    for this evaluator by appending _2, _3, … if needed.

    Args:
        t1_filename:   The T1 upload filename (used as the primary source).
        existing_refs: List of case_ref strings already in the evaluator's DB.

    Returns:
        A unique case ref string.
    """
    base = derive_case_ref(t1_filename)
    if not existing_refs or base not in existing_refs:
        return base

    for i in range(2, 100):
        candidate = f"{base}_{i}"
        if candidate not in existing_refs:
            return candidate

    # Extremely unlikely; fall back to timestamp suffix
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"{base}_{ts}"
