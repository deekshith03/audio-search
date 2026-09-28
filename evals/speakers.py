"""
Resolves a search result's speaker to the human name used by the qrels.

The pipeline emits anonymous diarization labels (SPEAKER_00 / SPEAKER_01); names come from the
human labeling step and live in dataset/speaker_labels/{file}.json.
"""

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional

LABELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "speaker_labels")
_ANONYMOUS_LABEL = re.compile(r"^SPEAKER_\d+$")


@lru_cache(maxsize=None)
def _speaker_names(file_id: str, labels_dir: str = LABELS_DIR) -> Dict[str, str]:
    path = os.path.join(labels_dir, f"{os.path.splitext(file_id)[0]}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("labels", {})


def resolve_result_speaker(result: Dict[str, Any], labels_dir: str = LABELS_DIR) -> Optional[str]:
    """Returns the human speaker name for a search result, resolving anonymous diarization labels."""
    speaker = result.get("speaker") or result.get("speaker_label")
    if speaker and _ANONYMOUS_LABEL.match(speaker):
        return _speaker_names(result.get("file_id", ""), labels_dir).get(speaker, speaker)
    return speaker
