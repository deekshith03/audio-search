"""
Human speaker labels: maps anonymous diarization labels (SPEAKER_00 / SPEAKER_01) to names.

Stored per file at `<workspace>/speaker_labels/{file_base}.json`, separate from the canonical
transcript, so renaming a speaker never requires re-processing or re-indexing.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.pipeline.common import Workspace, file_base, write_json

MAX_NAME_LENGTH = 80
LABELED_BY_VALUES = {"user", "golden_simulated"}


class LabelValidationError(ValueError):
    pass


def labels_path(workspace: Workspace, file_id: str) -> str:
    return os.path.join(workspace.labels_dir, f"{file_base(file_id)}.json")


def validate_labels(labels: Dict[str, str], speaker_labels: List[str]) -> Dict[str, str]:
    if set(labels) != set(speaker_labels):
        raise LabelValidationError(f"Expected names for exactly {sorted(speaker_labels)}, got {sorted(labels)}.")

    cleaned = {}
    for label in speaker_labels:
        name = " ".join((labels[label] or "").split())
        if not name:
            raise LabelValidationError(f"Name for {label} is empty.")
        if len(name) > MAX_NAME_LENGTH:
            raise LabelValidationError(f"Name for {label} is longer than {MAX_NAME_LENGTH} characters.")
        cleaned[label] = name

    if len({n.casefold() for n in cleaned.values()}) != len(cleaned):
        raise LabelValidationError("Each speaker needs a different name.")
    return cleaned


def save_labels(
    workspace: Workspace,
    file_id: str,
    labels: Dict[str, str],
    speaker_labels: List[str],
    labeled_by: str = "user",
) -> Dict[str, Any]:
    if labeled_by not in LABELED_BY_VALUES:
        raise LabelValidationError(f"labeled_by must be one of {sorted(LABELED_BY_VALUES)}.")
    doc = {
        "file_id": file_id,
        "labeled_by": labeled_by,
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": validate_labels(labels, speaker_labels),
    }
    write_json(labels_path(workspace, file_id), doc)
    return doc


def load_labels(workspace: Workspace, file_id: str) -> Optional[Dict[str, Any]]:
    path = labels_path(workspace, file_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def swap_labels(labels: Dict[str, str]) -> Dict[str, str]:
    if len(labels) != 2:
        raise LabelValidationError("Swap is only defined for two speakers.")
    (a, name_a), (b, name_b) = labels.items()
    return {a: name_b, b: name_a}


def speaker_name(labels_doc: Optional[Dict[str, Any]], speaker_label: str) -> str:
    if labels_doc:
        return labels_doc["labels"].get(speaker_label, speaker_label)
    return speaker_label
