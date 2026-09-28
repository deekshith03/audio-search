"""
Stage 4: Pipeline Quality Evaluation Suite (WER, Raw DER, Reconciled DER & True Word Speaker Accuracy).

Evaluates the generated canonical transcripts against Phase 1 ground truth:
1. WER (Word Error Rate via jiwer) with standard lower/punctuation normalization.
2. Raw Diarization DER (0ms & 250ms collar) from PyAnnote raw RTTM vs ground-truth intervals.
3. Reconciled Transcript DER (0ms & 250ms collar) from canonical word-bounded turns vs ground-truth intervals.
4. True Word Speaker Accuracy: Aligns reference and hypothesis token sequences via difflib SequenceMatcher
   and computes the speaker attribution accuracy over matched lexical words, along with coverage telemetry.
5. Generates run manifest (pipeline_manifest.json) recording SHA-256 hashes, versions, and quality scorecards.
"""

import os
import sys
import json
import re
import difflib
import hashlib
import itertools
from typing import Dict, Any, List, Tuple, Optional

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

_WHISPER_NORMALIZER = EnglishTextNormalizer()

# Predeclared evaluation scoring lexicon for domain compound equivalence
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
    """Basic lower/punctuation normalization for legacy baseline comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_text_openai(text: str) -> str:
    """Standard OpenAI Whisper text normalization with domain compound equivalence."""
    # Standardize unicode quotes so the OpenAI normalizer expands contractions cleanly
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = _WHISPER_NORMALIZER(text)
    for pat, rep in DOMAIN_COMPOUNDS.items():
        text = re.sub(pat, rep, text)
    return " ".join(text.split())


normalize_text = normalize_text_openai


def extract_normalized_tokens(text: str) -> List[str]:
    """Tokenizes text into normalized lexical word tokens."""
    return re.findall(r"\b[\w\']+\b", text.lower())


def compute_file_sha256(path: str) -> str:
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_rttm(rttm_path: str, uri: str) -> Annotation:
    """Parses an RTTM file into a PyAnnote Annotation object."""
    annot = Annotation(uri=uri)
    if not os.path.exists(rttm_path):
        return annot
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                tbeg = float(parts[3])
                tdur = float(parts[4])
                spkr = parts[7]
                annot[Segment(tbeg, tbeg + tdur)] = spkr
    return annot


def compute_evaluation_speaker_mapping(gt_data: Dict[str, Any], pipe_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Computes optimal Hungarian mapping between anonymous pipeline speakers and ground truth reference
    speakers based on temporal overlap strictly at evaluation time (Zero Data Leakage).
    """
    ref_turns = gt_data.get("turns", [])
    pipe_turns = pipe_data.get("turns", [])

    p_speakers = sorted(list({t.get("speaker_name", t.get("speaker_label", "")) for t in pipe_turns}))
    r_speakers = sorted(list({t.get("speaker", "") for t in ref_turns}))

    if not p_speakers or not r_speakers:
        return {}

    overlap_matrix = {p: {r: 0.0 for r in r_speakers} for p in p_speakers}
    for pt in pipe_turns:
        p_spk = pt.get("speaker_name", pt.get("speaker_label", ""))
        p_st = float(pt.get("start_seconds", 0.0))
        p_et = float(pt.get("end_seconds", 0.0))
        for rt in ref_turns:
            r_spk = str(rt.get("speaker") or "")
            r_st = float(rt.get("start_time", 0.0))
            r_et = float(rt.get("end_time", 0.0))
            ov = max(0.0, min(p_et, r_et) - max(p_st, r_st))
            if ov > 0:
                overlap_matrix[p_spk][r_spk] += ov
            elif p_spk == r_spk:
                # Identity fallback when timestamps are absent
                overlap_matrix[p_spk][r_spk] += 1.0

    best_mapping: Dict[str, str] = {}
    best_score = -1.0
    if len(r_speakers) >= len(p_speakers):
        for perm in itertools.permutations(r_speakers, len(p_speakers)):
            current = dict(zip(p_speakers, perm))
            score = sum(overlap_matrix[p][r] for p, r in current.items())
            if score > best_score:
                best_score = score
                best_mapping = current
    else:
        for p in p_speakers:
            best_ref = max(overlap_matrix[p].items(), key=lambda x: x[1])[0]
            best_mapping[p] = best_ref

    return best_mapping


def evaluate_word_speaker_accuracy(
    gt_data: Dict[str, Any],
    pipe_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes true word-level speaker accuracy via sequence alignment between
    reference word tokens and hypothesis recognized tokens using Hungarian mapping.
    """
    mapping = compute_evaluation_speaker_mapping(gt_data, pipe_data)

    ref_tokens: List[Tuple[str, str]] = []
    for t in gt_data.get("turns", []):
        spk = str(t.get("speaker") or "")
        for w in extract_normalized_tokens(t.get("text", "")):
            ref_tokens.append((w, spk))

    hyp_tokens: List[Tuple[str, str]] = []
    for t in pipe_data.get("turns", []):
        raw_spk = t.get("speaker_name", t.get("speaker_label", ""))
        mapped_spk = mapping.get(raw_spk, raw_spk)
        for w_obj in t.get("words", []):
            token_list = extract_normalized_tokens(w_obj.get("word", ""))
            if token_list:
                hyp_tokens.append((token_list[0], mapped_spk))

    ref_words = [t[0] for t in ref_tokens]
    hyp_words = [t[0] for t in hyp_tokens]

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matched_words = 0
    correct_speaker_words = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for r_idx, h_idx in zip(range(i1, i2), range(j1, j2)):
                matched_words += 1
                if ref_tokens[r_idx][1] == hyp_tokens[h_idx][1]:
                    correct_speaker_words += 1

    total_ref = len(ref_tokens)
    total_hyp = len(hyp_tokens)
    acc = (correct_speaker_words / matched_words) if matched_words > 0 else 0.0
    coverage = (matched_words / total_ref) if total_ref > 0 else 0.0

    return {
        "word_speaker_accuracy": round(acc, 4),
        "lexical_coverage": round(coverage, 4),
        "matched_words": matched_words,
        "correct_speaker_words": correct_speaker_words,
        "total_reference_words": total_ref,
        "total_hypothesis_words": total_hyp,
        "evaluation_speaker_mapping": mapping
    }


def evaluate_file_quality(
    canonical_path: str,
    gt_path: str,
    raw_rttm_path: str,
    audio_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates WER, Raw DER, Reconciled DER, and True Word Speaker Accuracy for a single file.
    """
    with open(canonical_path, "r", encoding="utf-8") as f:
        pipe_data = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    file_id = pipe_data["file_id"]

    # 1. Compute Full-Text WER (both Basic legacy and OpenAI Standard)
    ref_full_text = " ".join(t["text"] for t in gt_data["turns"])
    hyp_full_text = " ".join(t["text"] for t in pipe_data["turns"])

    b_norm_ref = normalize_text_basic(ref_full_text)
    b_norm_hyp = normalize_text_basic(hyp_full_text)
    basic_wer_score = float(jiwer.wer(b_norm_ref, b_norm_hyp))

    s_norm_ref = normalize_text_openai(ref_full_text)
    s_norm_hyp = normalize_text_openai(hyp_full_text)
    openai_wer_score = float(jiwer.wer(s_norm_ref, s_norm_hyp))

    # 2. Build Reference pyannote Annotation for DER
    ref_annot = Annotation(uri=file_id)
    for t in gt_data["turns"]:
        st = float(t["start_time"])
        et = float(t["end_time"])
        spk = t["speaker"]
        ref_annot[Segment(st, et)] = spk

    # Build Hypothesis pyannote Annotation from reconciled turns
    reconciled_hyp_annot = Annotation(uri=file_id)
    for t in pipe_data["turns"]:
        st = float(t["start_seconds"])
        et = float(t["end_seconds"])
        spk = t["speaker_label"]
        reconciled_hyp_annot[Segment(st, et)] = spk

    # Load Raw RTTM pyannote Annotation
    raw_hyp_annot = parse_rttm(raw_rttm_path, file_id)

    # Initialize DER metrics
    der_metric_strict = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der_metric_forgiving = DiarizationErrorRate(collar=0.25, skip_overlap=False)

    # Raw PyAnnote DER with component breakdown (Confusion, Missed Detection, False Alarm)
    raw_details = der_metric_strict(ref_annot, raw_hyp_annot, detailed=True)
    raw_tot = float(raw_details.get("total", 0.0))
    raw_conf = float(raw_details.get("confusion", 0.0))
    raw_miss = float(raw_details.get("missed detection", 0.0))
    raw_fa = float(raw_details.get("false alarm", 0.0))
    raw_der_0ms = float(raw_details.get("diarization error rate", 0.0))

    # Conditioned on speech activity: what fraction of detected speech had wrong speaker?
    detected_speech = raw_tot - raw_miss
    speaker_attribution_error = (raw_conf / detected_speech) if detected_speech > 0 else 0.0

    raw_forgiving_details = der_metric_forgiving(ref_annot, raw_hyp_annot, detailed=True)
    raw_der_250ms = float(raw_forgiving_details.get("diarization error rate", 0.0))

    # Reconciled canonical turns DER
    rec_der_0ms_res = der_metric_strict(ref_annot, reconciled_hyp_annot)
    rec_der_0ms = float(rec_der_0ms_res if isinstance(rec_der_0ms_res, (int, float)) else getattr(rec_der_0ms_res, "error", 0.0))

    rec_der_250ms_res = der_metric_forgiving(ref_annot, reconciled_hyp_annot)
    rec_der_250ms = float(rec_der_250ms_res if isinstance(rec_der_250ms_res, (int, float)) else getattr(rec_der_250ms_res, "error", 0.0))

    # 3. Compute True Word Speaker Accuracy
    spk_eval = evaluate_word_speaker_accuracy(gt_data, pipe_data)

    # 4. Hashes
    audio_sha = compute_file_sha256(audio_path) if audio_path else ""
    canonical_sha = compute_file_sha256(canonical_path)
    rttm_sha = compute_file_sha256(raw_rttm_path)

    return {
        "file_id": file_id,
        "audio_sha256": audio_sha,
        "canonical_sha256": canonical_sha,
        "raw_rttm_sha256": rttm_sha,
        "wer": round(openai_wer_score, 4),
        "standard_wer": round(openai_wer_score, 4),
        "basic_wer": round(basic_wer_score, 4),
        "raw_der_0ms_collar": round(raw_der_0ms, 4),
        "raw_der_250ms_collar": round(raw_der_250ms, 4),
        "raw_confusion_rate": round(raw_conf / raw_tot, 4) if raw_tot > 0 else 0.0,
        "raw_missed_detection_rate": round(raw_miss / raw_tot, 4) if raw_tot > 0 else 0.0,
        "raw_false_alarm_rate": round(raw_fa / raw_tot, 4) if raw_tot > 0 else 0.0,
        "speaker_attribution_error": round(speaker_attribution_error, 4),
        "confusion_seconds": round(raw_conf, 2),
        "missed_seconds": round(raw_miss, 2),
        "false_alarm_seconds": round(raw_fa, 2),
        "total_speech_seconds": round(raw_tot, 2),
        "reconciled_der_0ms_collar": round(rec_der_0ms, 4),
        "reconciled_der_250ms_collar": round(rec_der_250ms, 4),
        "word_speaker_accuracy": spk_eval["word_speaker_accuracy"],
        "lexical_coverage": spk_eval["lexical_coverage"],
        "matched_words": spk_eval["matched_words"],
        "correct_speaker_words": spk_eval["correct_speaker_words"],
        "total_reference_words": spk_eval["total_reference_words"],
        "total_turns": len(pipe_data["turns"]),
        "audio_duration_seconds": pipe_data.get("audio_duration_seconds", 0.0),
        "telemetry": pipe_data.get("telemetry", {})
    }


def run_pipeline_evaluation(
    audio_dir: str = "dataset/audio",
    out_dir: str = "dataset/pipeline_outputs",
    gt_dir: str = "dataset/ground_truth",
    rttm_dir: str = "dataset/cache/raw_diarization"
) -> Dict[str, Any]:
    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    per_file_scores = []

    print("\n" + "=" * 120)
    print("                    PHASE 2 PIPELINE QUALITY SCORECARD (ASR & DIARIZATION AUDIT)")
    print("=" * 120)
    print(f"{'File ID':<35} | {'Std WER':<8} | {'Raw DER':<8} | {'Confusion':<9} | {'Miss(Pause)':<11} | {'FA':<5} | {'SpkAttrErr':<10} | {'Spk Acc':<7}")
    print("-" * 120)

    clean_wers = []
    clean_basic_wers = []
    clean_raw_ders_0 = []
    clean_raw_ders_250 = []
    clean_raw_confs = []
    clean_raw_misses = []
    clean_raw_fas = []
    clean_spk_attr_errs = []
    clean_rec_ders_0 = []
    clean_rec_ders_250 = []
    clean_spk_accs = []
    clean_coverages = []

    cafe_scores = None

    for af in files:
        base = af.replace(".wav", "")
        audio_p = os.path.join(audio_dir, af)
        canonical_p = os.path.join(out_dir, f"{base}_canonical.json")
        gt_p = os.path.join(gt_dir, f"{base}.json")
        rttm_p = os.path.join(rttm_dir, f"{base}.rttm")

        if not os.path.exists(canonical_p):
            print(f"[{af}] Canonical transcript not found: {canonical_p}")
            continue

        q = evaluate_file_quality(canonical_p, gt_p, rttm_p, audio_path=audio_p)
        per_file_scores.append(q)

        print(f"{q['file_id']:<35} | {q['standard_wer']:<8.2%} | {q['raw_der_0ms_collar']:<8.2%} | {q['raw_confusion_rate']:<9.2%} | {q['raw_missed_detection_rate']:<11.2%} | {q['raw_false_alarm_rate']:<5.2%} | {q['speaker_attribution_error']:<10.2%} | {q['word_speaker_accuracy']:<7.2%}")

        if "cafe" in af:
            cafe_scores = q
        else:
            clean_wers.append(q["standard_wer"])
            clean_basic_wers.append(q["basic_wer"])
            clean_raw_ders_0.append(q["raw_der_0ms_collar"])
            clean_raw_ders_250.append(q["raw_der_250ms_collar"])
            clean_raw_confs.append(q["raw_confusion_rate"])
            clean_raw_misses.append(q["raw_missed_detection_rate"])
            clean_raw_fas.append(q["raw_false_alarm_rate"])
            clean_spk_attr_errs.append(q["speaker_attribution_error"])
            clean_rec_ders_0.append(q["reconciled_der_0ms_collar"])
            clean_rec_ders_250.append(q["reconciled_der_250ms_collar"])
            clean_spk_accs.append(q["word_speaker_accuracy"])
            clean_coverages.append(q["lexical_coverage"])

    print("=" * 120)

    def avg(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    print("\n[PERFORMANCE BREAKDOWN: CLEAN FILES AVERAGE (audio_01 - 06)]")
    if clean_wers:
        print(f"  • Standardized WER (OpenAI Normalizer):   {avg(clean_wers):.2%}")
        print(f"  • Basic WER (Legacy lower/punct):         {avg(clean_basic_wers):.2%}")
        print(f"  • Raw PyAnnote DER (0ms strict):          {avg(clean_raw_ders_0):.2%}")
        print(f"      ├── Speaker Confusion Rate:           {avg(clean_raw_confs):.2%} (True identity mismatch)")
        print(f"      ├── Missed Detection (Silence Pauses):{avg(clean_raw_misses):.2%} (Artifact of 0s continuous GT)")
        print(f"      └── False Alarm Rate:                 {avg(clean_raw_fas):.2%}")
        print(f"  • Pure Speaker Attribution Error (SAD):   {avg(clean_spk_attr_errs):.2%} (Error when speech present)")
        print(f"  • Raw PyAnnote DER (250ms collar):        {avg(clean_raw_ders_250):.2%}")
        print(f"  • Reconciled Transcript DER (0ms strict): {avg(clean_rec_ders_0):.2%}")
        print(f"  • Reconciled Transcript DER (250ms collar):{avg(clean_rec_ders_250):.2%}")
        print(f"  • True Word Speaker Accuracy:             {avg(clean_spk_accs):.2%}")
        print(f"  • Lexical Alignment Coverage:             {avg(clean_coverages):.2%}")

    if cafe_scores:
        print("\n[NOISY CAFE FILE BREAKDOWN (audio_07 - 12 dB SNR)]")
        print(f"  • Standardized WER (OpenAI Normalizer):   {cafe_scores['standard_wer']:.2%}")
        print(f"  • Basic WER (Legacy lower/punct):         {cafe_scores['basic_wer']:.2%}")
        print(f"  • Raw PyAnnote DER (0ms strict):          {cafe_scores['raw_der_0ms_collar']:.2%}")
        print(f"      ├── Speaker Confusion Rate:           {cafe_scores['raw_confusion_rate']:.2%}")
        print(f"      ├── Missed Detection (Silence Pauses):{cafe_scores['raw_missed_detection_rate']:.2%}")
        print(f"      └── False Alarm Rate:                 {cafe_scores['raw_false_alarm_rate']:.2%}")
        print(f"  • Pure Speaker Attribution Error (SAD):   {cafe_scores['speaker_attribution_error']:.2%}")
        print(f"  • Raw PyAnnote DER (250ms collar):        {cafe_scores['raw_der_250ms_collar']:.2%}")
        print(f"  • Reconciled Transcript DER (0ms strict): {cafe_scores['reconciled_der_0ms_collar']:.2%}")
        print(f"  • Reconciled Transcript DER (250ms collar):{cafe_scores['reconciled_der_250ms_collar']:.2%}")
        print(f"  • True Word Speaker Accuracy:             {cafe_scores['word_speaker_accuracy']:.2%}")
        print(f"  • Lexical Alignment Coverage:             {cafe_scores['lexical_coverage']:.2%}")

    # Build pipeline run manifest with complete hashes and audit metrics
    manifest = {
        "pipeline_version": "2.0.0",
        "models": {
            "asr_primary": "mlx-community/whisper-large-v3-turbo",
            "alignment": "wav2vec2-large-960h",
            "diarizer": "pyannote/speaker-diarization-community-1",
            "asr_ablation_status": "deferred"
        },
        "clean_files_summary": {
            "mean_standard_wer": round(avg(clean_wers), 4),
            "mean_basic_wer": round(avg(clean_basic_wers), 4),
            "mean_raw_der_0ms": round(avg(clean_raw_ders_0), 4),
            "mean_raw_der_250ms": round(avg(clean_raw_ders_250), 4),
            "mean_raw_confusion_rate": round(avg(clean_raw_confs), 4),
            "mean_raw_missed_detection_rate": round(avg(clean_raw_misses), 4),
            "mean_raw_false_alarm_rate": round(avg(clean_raw_fas), 4),
            "mean_speaker_attribution_error": round(avg(clean_spk_attr_errs), 4),
            "mean_reconciled_der_0ms": round(avg(clean_rec_ders_0), 4),
            "mean_reconciled_der_250ms": round(avg(clean_rec_ders_250), 4),
            "mean_word_speaker_accuracy": round(avg(clean_spk_accs), 4),
            "mean_lexical_coverage": round(avg(clean_coverages), 4)
        },
        "cafe_file_summary": cafe_scores,
        "files": per_file_scores
    }

    manifest_path = os.path.join(out_dir, "pipeline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved audit-grade pipeline manifest to {manifest_path}\n")
    return manifest


if __name__ == "__main__":
    run_pipeline_evaluation()
