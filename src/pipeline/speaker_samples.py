"""
Representative audio snippets per diarized speaker, for the human labeling step.

For each speaker, candidate windows are cut from that speaker's turns at word boundaries
(4-15 s by default), scored by mean word confidence, and chosen so the picks are spread across
the recording (best candidate per equal time slice first, then best remaining overall). Turns too
short to reach the minimum are used only when a speaker has nothing longer.
"""

import io
from typing import Any, Dict, List

import soundfile as sf

MIN_SNIPPET_SECONDS = 4.0
MAX_SNIPPET_SECONDS = 15.0
FALLBACK_MIN_SECONDS = 1.0


def _window_from_turn(turn: Dict[str, Any], max_seconds: float) -> Dict[str, Any]:
    words = turn["words"]
    start = words[0]["start_seconds"]
    picked = [w for w in words if w["end_seconds"] - start <= max_seconds] or words[:1]
    confidences = [w["confidence"] for w in picked if w.get("confidence") is not None]
    return {
        "speaker_label": turn["speaker_label"],
        "turn_id": turn["turn_id"],
        "start_seconds": round(start, 3),
        "end_seconds": round(picked[-1]["end_seconds"], 3),
        "text": " ".join(w["word"] for w in picked),
        "score": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def _candidates(turns: List[Dict[str, Any]], speaker: str, min_seconds: float, max_seconds: float) -> List[Dict[str, Any]]:
    out = []
    for t in turns:
        if t["speaker_label"] != speaker or t.get("is_short_turn") or not t["words"]:
            continue
        window = _window_from_turn(t, max_seconds)
        if window["end_seconds"] - window["start_seconds"] >= min_seconds:
            out.append(window)
    return out


def _spread_pick(candidates: List[Dict[str, Any]], n: int, duration: float) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    slice_len = max(duration, 1e-6) / n
    picked: List[Dict[str, Any]] = []
    for i in range(n):
        in_slice = [c for c in candidates if i * slice_len <= c["start_seconds"] < (i + 1) * slice_len and c not in picked]
        if in_slice:
            picked.append(max(in_slice, key=lambda c: c["score"]))
    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if len(picked) >= n:
            break
        if c not in picked:
            picked.append(c)
    return sorted(picked, key=lambda c: c["start_seconds"])


def select_speaker_samples(
    canonical: Dict[str, Any],
    per_speaker: int = 3,
    min_seconds: float = MIN_SNIPPET_SECONDS,
    max_seconds: float = MAX_SNIPPET_SECONDS,
) -> Dict[str, List[Dict[str, Any]]]:
    turns = canonical["turns"]
    duration = float(canonical.get("audio_duration_seconds") or max(t["end_seconds"] for t in turns))
    samples = {}
    for speaker in canonical["speaker_labels"]:
        candidates = _candidates(turns, speaker, min_seconds, max_seconds)
        if len(candidates) < per_speaker:
            longest_first = sorted(
                _candidates(turns, speaker, FALLBACK_MIN_SECONDS, max_seconds),
                key=lambda c: c["end_seconds"] - c["start_seconds"],
                reverse=True,
            )
            candidates += [c for c in longest_first if c not in candidates][: per_speaker - len(candidates)]
        samples[speaker] = _spread_pick(candidates, per_speaker, duration)
    return samples


def speaking_time(canonical: Dict[str, Any]) -> Dict[str, float]:
    totals = {s: 0.0 for s in canonical["speaker_labels"]}
    for t in canonical["turns"]:
        totals[t["speaker_label"]] = totals.get(t["speaker_label"], 0.0) + t["end_seconds"] - t["start_seconds"]
    return {s: round(v, 1) for s, v in totals.items()}


def extract_clip(wav_path: str, start: float, end: float, pad_seconds: float = 0.15) -> bytes:
    info = sf.info(wav_path)
    s = max(0.0, start - pad_seconds)
    e = min(info.duration, end + pad_seconds)
    data, sr = sf.read(wav_path, start=int(s * info.samplerate), stop=int(e * info.samplerate), dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
