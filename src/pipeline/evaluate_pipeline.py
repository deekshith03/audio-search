"""
Stage 4: Pipeline Quality Evaluation (golden set only).

Scores canonical transcripts against the Phase 1 ground truth:
1. WER (jiwer) with the OpenAI Whisper English normalizer plus domain compound equivalence,
   and a legacy lower/punctuation-only "basic" WER for comparison.
2. DER (pyannote.metrics, 0 ms and 250 ms collars, overlap scored) at three points:
   - raw:        pyannote's untouched overlap-aware output ({id}.raw.rttm)
   - refined:    exclusive intervals after centroid refinement ({id}.rttm)
   - reconciled: word-bounded canonical turns
   The reference turns are gap-free (silences are labelled as speech), so every system is
   charged "missed detection" for real pauses. Reconciled turns bridge pauses up to 1.5 s and
   therefore score lower DER mainly because they match that gap-free convention, not because
   speaker attribution improved.
3. Word speaker accuracy: aligns reference and hypothesis tokens (difflib) and measures the
   share of matched words attributed to the correct speaker under the optimal label mapping.
   Timestamp-independent, so it is the headline speaker metric.
4. Writes pipeline_manifest.json with model provenance, input/output SHA-256 hashes and scores.
"""

import itertools
import json
import os
import re
import difflib
from typing import Any, Dict, List, Tuple

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

from src.pipeline.common import (
    DIARIZATION_DIR,
    GROUND_TRUTH_DIR,
    OUTPUT_DIR,
    RAW_ASR_DIR,
    file_base,
    parse_stage_args,
    resolve_audio_files,
    sha256_file,
    write_json,
)

_WHISPER_NORMALIZER = EnglishTextNormalizer()

DOMAIN_COMPOUNDS = {
    r"\bpg\s+vector\b": "pgvector",
    r"\bpg\s+mustard\b": "pgmustard",
    r"\bhypr\s+land\b": "hyprland",
    r"\bhyper\s+land\b": "hyprland",
    r"\bco\s+wos\b": "cowos",
    r"\bcap\s+ex\b": "capex",
    r"\bopen\s+ai\b": "openai",
}


def normalize_text_basic(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_text_openai(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = _WHISPER_NORMALIZER(text)
    for pat, rep in DOMAIN_COMPOUNDS.items():
        text = re.sub(pat, rep, text)
    return " ".join(text.split())


normalize_text = normalize_text_openai


def extract_normalized_tokens(text: str) -> List[str]:
    return re.findall(r"\b[\w\']+\b", text.lower())


def parse_rttm(rttm_path: str, uri: str) -> Annotation:
    annot = Annotation(uri=uri)
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                tbeg = float(parts[3])
                annot[Segment(tbeg, tbeg + float(parts[4]))] = parts[7]
    return annot


def compute_evaluation_speaker_mapping(gt_data: Dict[str, Any], pipe_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Optimal one-to-one mapping from anonymous pipeline labels to reference speakers by total
    temporal overlap. Used only inside scoring, never written back to pipeline outputs.
    """
    ref_turns = gt_data.get("turns", [])
    pipe_turns = pipe_data.get("turns", [])
    p_speakers = sorted({t["speaker_label"] for t in pipe_turns})
    r_speakers = sorted({t["speaker"] for t in ref_turns})
    if not p_speakers or not r_speakers:
        return {}

    overlap = {p: {r: 0.0 for r in r_speakers} for p in p_speakers}
    for pt in pipe_turns:
        for rt in ref_turns:
            ov = min(pt["end_seconds"], rt["end_time"]) - max(pt["start_seconds"], rt["start_time"])
            if ov > 0:
                overlap[pt["speaker_label"]][rt["speaker"]] += ov

    if len(r_speakers) < len(p_speakers):
        return {p: max(overlap[p].items(), key=lambda x: x[1])[0] for p in p_speakers}

    best_mapping: Dict[str, str] = {}
    best_score = -1.0
    for perm in itertools.permutations(r_speakers, len(p_speakers)):
        candidate = dict(zip(p_speakers, perm))
        score = sum(overlap[p][r] for p, r in candidate.items())
        if score > best_score:
            best_score, best_mapping = score, candidate
    return best_mapping


def evaluate_word_speaker_accuracy(gt_data: Dict[str, Any], pipe_data: Dict[str, Any]) -> Dict[str, Any]:
    mapping = compute_evaluation_speaker_mapping(gt_data, pipe_data)

    ref_tokens: List[Tuple[str, str]] = [
        (w, t["speaker"]) for t in gt_data.get("turns", []) for w in extract_normalized_tokens(t.get("text", ""))
    ]
    hyp_tokens: List[Tuple[str, str]] = []
    for t in pipe_data.get("turns", []):
        spk = mapping.get(t["speaker_label"], t["speaker_label"])
        for w_obj in t.get("words", []):
            hyp_tokens.extend((tok, spk) for tok in extract_normalized_tokens(w_obj.get("word", "")))

    matcher = difflib.SequenceMatcher(None, [t[0] for t in ref_tokens], [t[0] for t in hyp_tokens], autojunk=False)
    matched = 0
    correct = 0
    for tag, i1, i2, j1, _ in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                matched += 1
                correct += ref_tokens[i1 + k][1] == hyp_tokens[j1 + k][1]

    return {
        "word_speaker_accuracy": round(correct / matched, 4) if matched else 0.0,
        "lexical_coverage": round(matched / len(ref_tokens), 4) if ref_tokens else 0.0,
        "matched_words": matched,
        "correct_speaker_words": correct,
        "total_reference_words": len(ref_tokens),
        "total_hypothesis_words": len(hyp_tokens),
        "evaluation_speaker_mapping": mapping,
    }


def _der(reference: Annotation, hypothesis: Annotation) -> Dict[str, float]:
    strict = DiarizationErrorRate(collar=0.0, skip_overlap=False)(reference, hypothesis, detailed=True)
    forgiving = DiarizationErrorRate(collar=0.25, skip_overlap=False)(reference, hypothesis)
    total = float(strict["total"])
    miss = float(strict["missed detection"])
    conf = float(strict["confusion"])
    fa = float(strict["false alarm"])
    detected = total - miss
    return {
        "der_0ms": round(float(strict["diarization error rate"]), 4),
        "der_250ms": round(float(forgiving), 4),
        "confusion_rate": round(conf / total, 4) if total else 0.0,
        "missed_detection_rate": round(miss / total, 4) if total else 0.0,
        "false_alarm_rate": round(fa / total, 4) if total else 0.0,
        "speaker_attribution_error": round(conf / detected, 4) if detected > 0 else 0.0,
        "confusion_seconds": round(conf, 2),
        "missed_seconds": round(miss, 2),
        "false_alarm_seconds": round(fa, 2),
        "total_speech_seconds": round(total, 2),
    }


def evaluate_file_quality(audio_path: str, gt_path: str) -> Dict[str, Any]:
    base = file_base(audio_path)
    canonical_path = os.path.join(OUTPUT_DIR, f"{base}_canonical.json")
    raw_rttm_path = os.path.join(DIARIZATION_DIR, f"{base}.raw.rttm")
    refined_rttm_path = os.path.join(DIARIZATION_DIR, f"{base}.rttm")
    diar_json_path = os.path.join(DIARIZATION_DIR, f"{base}_diarization.json")
    raw_asr_path = os.path.join(RAW_ASR_DIR, f"{base}_raw.json")

    with open(canonical_path, "r", encoding="utf-8") as f:
        pipe_data = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(diar_json_path, "r", encoding="utf-8") as f:
        diar_data = json.load(f)
    with open(raw_asr_path, "r", encoding="utf-8") as f:
        raw_asr = json.load(f)

    file_id = pipe_data["file_id"]
    ref_text = " ".join(t["text"] for t in gt_data["turns"])
    hyp_text = " ".join(t["text"] for t in pipe_data["turns"])
    wer_std = jiwer.process_words(normalize_text_openai(ref_text), normalize_text_openai(hyp_text))
    wer_basic = jiwer.wer(normalize_text_basic(ref_text), normalize_text_basic(hyp_text))

    ref_annot = Annotation(uri=file_id)
    for t in gt_data["turns"]:
        ref_annot[Segment(float(t["start_time"]), float(t["end_time"]))] = t["speaker"]
    reconciled_annot = Annotation(uri=file_id)
    for t in pipe_data["turns"]:
        reconciled_annot[Segment(float(t["start_seconds"]), float(t["end_seconds"]))] = t["speaker_label"]

    spk_eval = evaluate_word_speaker_accuracy(gt_data, pipe_data)

    return {
        "file_id": file_id,
        "hashes": {
            "audio_sha256": sha256_file(audio_path),
            "raw_rttm_sha256": sha256_file(raw_rttm_path),
            "refined_rttm_sha256": sha256_file(refined_rttm_path),
            "canonical_sha256": sha256_file(canonical_path),
        },
        "runtime_seconds": {
            "asr": raw_asr.get("runtime_seconds"),
            "diarization": diar_data.get("runtime_seconds"),
        },
        "standard_wer": round(float(wer_std.wer), 4),
        "wer_breakdown": {"substitutions": wer_std.substitutions, "deletions": wer_std.deletions, "insertions": wer_std.insertions},
        "basic_wer": round(float(wer_basic), 4),
        "der_raw": _der(ref_annot, parse_rttm(raw_rttm_path, file_id)),
        "der_refined": _der(ref_annot, parse_rttm(refined_rttm_path, file_id)),
        "der_reconciled": _der(ref_annot, reconciled_annot),
        "diarization_refinement": diar_data.get("refinement", {}),
        **{k: spk_eval[k] for k in ("word_speaker_accuracy", "lexical_coverage", "matched_words", "correct_speaker_words", "total_reference_words")},
        "total_turns": len(pipe_data["turns"]),
        "audio_duration_seconds": pipe_data["audio_duration_seconds"],
        "telemetry": pipe_data.get("telemetry", {}),
    }


def _mean(rows: List[Dict[str, Any]], getter) -> float:
    vals = [getter(r) for r in rows]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "files": len(rows),
        "mean_standard_wer": _mean(rows, lambda r: r["standard_wer"]),
        "mean_basic_wer": _mean(rows, lambda r: r["basic_wer"]),
        "mean_raw_der_0ms": _mean(rows, lambda r: r["der_raw"]["der_0ms"]),
        "mean_raw_der_250ms": _mean(rows, lambda r: r["der_raw"]["der_250ms"]),
        "mean_raw_confusion_rate": _mean(rows, lambda r: r["der_raw"]["confusion_rate"]),
        "mean_raw_missed_detection_rate": _mean(rows, lambda r: r["der_raw"]["missed_detection_rate"]),
        "mean_raw_false_alarm_rate": _mean(rows, lambda r: r["der_raw"]["false_alarm_rate"]),
        "mean_raw_speaker_attribution_error": _mean(rows, lambda r: r["der_raw"]["speaker_attribution_error"]),
        "mean_refined_der_0ms": _mean(rows, lambda r: r["der_refined"]["der_0ms"]),
        "mean_refined_der_250ms": _mean(rows, lambda r: r["der_refined"]["der_250ms"]),
        "mean_reconciled_der_0ms": _mean(rows, lambda r: r["der_reconciled"]["der_0ms"]),
        "mean_reconciled_der_250ms": _mean(rows, lambda r: r["der_reconciled"]["der_250ms"]),
        "mean_word_speaker_accuracy": _mean(rows, lambda r: r["word_speaker_accuracy"]),
        "mean_lexical_coverage": _mean(rows, lambda r: r["lexical_coverage"]),
    }


def _load_stage_configs(base: str) -> Dict[str, Any]:
    configs = {}
    for stage, path in (
        ("asr", os.path.join(RAW_ASR_DIR, f"{base}_raw.json")),
        ("align", os.path.join(RAW_ASR_DIR, f"{base}_aligned.json")),
        ("diarize", os.path.join(DIARIZATION_DIR, f"{base}_diarization.json")),
    ):
        with open(path, "r", encoding="utf-8") as f:
            configs[stage] = json.load(f).get("config")
    return configs


def run_pipeline_evaluation(audio_files: List[str], gt_dir: str = GROUND_TRUTH_DIR) -> Dict[str, Any]:
    rows = []
    header = f"{'File':<40} | {'WER':>6} | {'RawDER':>6} | {'RefDER':>6} | {'RecDER':>6} | {'Conf':>5} | {'SpkAcc':>6}"
    print("\n" + "=" * len(header))
    print("PHASE 2 PIPELINE QUALITY SCORECARD (0 ms collar)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for audio_path in audio_files:
        base = file_base(audio_path)
        gt_path = os.path.join(gt_dir, f"{base}.json")
        if not os.path.exists(gt_path):
            print(f"[{base}] No ground truth; skipping evaluation.")
            continue
        q = evaluate_file_quality(audio_path, gt_path)
        rows.append(q)
        print(
            f"{base:<40} | {q['standard_wer']:>6.2%} | {q['der_raw']['der_0ms']:>6.2%} | "
            f"{q['der_refined']['der_0ms']:>6.2%} | {q['der_reconciled']['der_0ms']:>6.2%} | "
            f"{q['der_raw']['confusion_rate']:>5.2%} | {q['word_speaker_accuracy']:>6.2%}"
        )

    if not rows:
        raise SystemExit("No files with ground truth were evaluated.")

    clean = [r for r in rows if "cafe" not in r["file_id"]]
    noisy = [r for r in rows if "cafe" in r["file_id"]]
    summary = {"all_files": summarize(rows), "clean_files": summarize(clean), "noisy_files": summarize(noisy)}

    print("=" * len(header))
    for name, s in summary.items():
        if s["files"]:
            print(
                f"{name:<12} n={s['files']} | WER {s['mean_standard_wer']:.2%} | raw DER {s['mean_raw_der_0ms']:.2%} "
                f"| refined DER {s['mean_refined_der_0ms']:.2%} | reconciled DER {s['mean_reconciled_der_0ms']:.2%} "
                f"| word speaker acc {s['mean_word_speaker_accuracy']:.2%}"
            )

    first_base = file_base(rows[0]["file_id"])
    with open(os.path.join(OUTPUT_DIR, f"{first_base}_canonical.json"), "r", encoding="utf-8") as f:
        pipeline_version = json.load(f)["pipeline_version"]

    manifest = {
        "pipeline_version": pipeline_version,
        "stage_configs": _load_stage_configs(first_base),
        "summary": summary,
        "files": rows,
    }
    manifest_path = os.path.join(OUTPUT_DIR, "pipeline_manifest.json")
    write_json(manifest_path, manifest)
    print(f"\nSaved pipeline manifest to {manifest_path}")
    return manifest


if __name__ == "__main__":
    args = parse_stage_args("Stage 4: pipeline quality evaluation against ground truth")
    run_pipeline_evaluation(resolve_audio_files(args))
