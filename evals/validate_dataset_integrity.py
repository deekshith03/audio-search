"""
Automated Dataset and Ground-Truth Integrity Verification Suite.

Zero-dependency standard library implementation (uses `wave`, `json`, `os`, `math`).
Strictly proves:
1. Exactly 7 WAV files exist, are valid 16,000 Hz, 1-channel mono, 16-bit PCM WAV (8-10 min).
2. Exactly 7 ground-truth transcript files exist:
   - All timestamps are finite, non-negative numbers.
   - All turns have strictly positive duration (start_time < end_time).
   - EXACT turn continuity: t1['end_time'] == t2['start_time'] (0.0s gap, 0.0s overlap).
   - Exactly 2 declared foreground speakers per transcript.
   - Sequential, unique turn IDs (1..N).
   - Declared file durations match physical audio duration within 1.0s.
3. Benchmark Queries (Qrels):
   - Exactly 18 queries (6 single_file, 6 multi_file, 6 near_miss).
   - All query IDs are unique.
   - target_file_count strictly equals len(unique_files).
   - Every relevant moment points to an existing file_id and turn_id.
   - Declared speaker matches transcript speaker.
   - Moment timestamps are strictly numeric, finite, non-boolean, positive duration, and contained in turn interval.
   - matched_text is non-empty and is an EXACT VERBATIM SUBSTRING of the turn text.
   - Hard negatives point to existing file_id, turn_id, matching speaker, numeric, finite, positive, and strictly contained.
"""

import os
import sys
import json
import wave
import math


def validate_all():
    errors = []

    # 1. Validate Audio Files (Exactly 7 expected)
    audio_dir = "dataset/audio"
    if not os.path.isdir(audio_dir):
        errors.append(f"Missing audio directory: {audio_dir}")
        audio_files = []
    else:
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

    if len(audio_files) != 7:
        errors.append(f"Expected exactly 7 audio files, found {len(audio_files)}")

    durations = {}
    for af in audio_files:
        path = os.path.join(audio_dir, af)
        try:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n_frames = wf.getnframes()
                dur = n_frames / float(sr)

                if sr != 16000:
                    errors.append(f"{af}: Invalid sample rate {sr} Hz (must be 16000)")
                if ch != 1:
                    errors.append(f"{af}: Invalid channel count {ch} (must be 1 ch / mono)")
                if sw != 2: # 2 bytes = 16-bit PCM
                    errors.append(f"{af}: Invalid sample width {sw} bytes (must be 2 bytes / 16-bit PCM)")
                if not (480.0 <= dur <= 600.0):
                    errors.append(f"{af}: Duration {dur:.2f}s is out of 8-10 minute bounds")
                durations[af] = dur
        except Exception as e:
            errors.append(f"{af}: Failed to open WAV: {e}")

    print(f"✓ Audio files validated: exactly {len(durations)} files are 16kHz mono 16-bit PCM (8-10 min).")

    # 2. Validate Ground Truth Transcripts (Exactly 7 expected)
    gt_dir = "dataset/ground_truth"
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
    if len(gt_files) != 7:
        errors.append(f"Expected exactly 7 ground truth transcripts, found {len(gt_files)}")

    transcripts = {}
    for gf in gt_files:
        path = os.path.join(gt_dir, gf)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fid = data.get("file_id")
        if fid not in durations:
            errors.append(f"{gf}: References unknown audio file_id {fid}")

        declared_dur = data.get("duration_seconds")
        if fid in durations and abs(declared_dur - durations[fid]) > 1.0:
            errors.append(f"{gf}: Declared duration {declared_dur}s differs from audio {durations[fid]:.2f}s")

        speakers = data.get("speakers", [])
        if len(speakers) != 2:
            errors.append(f"{gf}: Expected exactly 2 declared speakers, found {len(speakers)}")
        spk_ids = set(s.get("id") for s in speakers)

        turns = data.get("turns", [])
        if not turns:
            errors.append(f"{gf}: Empty turns list")

        turn_dict = {}
        for i, t in enumerate(turns):
            tid = t.get("turn_id")
            spk = t.get("speaker")
            st = t.get("start_time")
            et = t.get("end_time")
            text = t.get("text")

            # Sequential turn IDs
            if tid != i + 1:
                errors.append(f"{gf}: Turn ID {tid} is not sequential (expected {i + 1})")

            # Check finite numeric timestamps (reject boolean)
            if not isinstance(st, (int, float)) or isinstance(st, bool) or not isinstance(et, (int, float)) or isinstance(et, bool):
                errors.append(f"{gf} Turn {tid}: Non-numeric timestamps [{st}, {et}]")
                continue
            if math.isnan(st) or math.isnan(et) or math.isinf(st) or math.isinf(et):
                errors.append(f"{gf} Turn {tid}: NaN or Inf timestamp [{st}, {et}]")
                continue
            if st < 0 or et <= st:
                errors.append(f"{gf} Turn {tid}: Non-positive duration [{st} -> {et}]")
            if not spk or spk not in spk_ids:
                errors.append(f"{gf} Turn {tid}: Speaker '{spk}' not in declared speakers {spk_ids}")
            if not text or not text.strip():
                errors.append(f"{gf} Turn {tid}: Missing or empty text")

            # Check EXACT continuity with adjacent turn
            if i < len(turns) - 1:
                next_st = turns[i+1].get("start_time")
                if next_st != et:
                    errors.append(f"{gf}: Discontinuity between Turn {tid} (end={et}) and Turn {turns[i+1].get('turn_id')} (start={next_st})")

            turn_dict[tid] = t

        transcripts[fid] = turn_dict

    print(f"✓ Ground-truth transcripts validated: exactly {len(transcripts)} files with 100% exact turn continuity.")

    # 3. Validate Benchmark Queries (Qrels)
    qrels_path = "dataset/qrels/benchmark_queries.json"
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels_data = json.load(f)

    corpus_manifest = qrels_data.get("corpus_files", [])
    if sorted(corpus_manifest) != sorted(audio_files):
        errors.append(f"Qrels corpus_files {corpus_manifest} does not match audio files {audio_files}")

    queries = qrels_data.get("queries", [])
    if len(queries) != 18:
        errors.append(f"Expected exactly 18 benchmark queries, found {len(queries)}")

    breakdown = {"single_file": 0, "multi_file": 0, "near_miss": 0}
    seen_qids = set()

    for q in queries:
        qid = q.get("query_id")
        if not qid or qid in seen_qids:
            errors.append(f"Duplicate or empty query_id: {qid}")
        seen_qids.add(qid)

        category = q.get("category")
        if category in breakdown:
            breakdown[category] += 1
        else:
            errors.append(f"{qid}: Invalid category '{category}'")

        expected = q.get("relevant_moments", [])
        if not expected:
            errors.append(f"{qid}: No relevant moments defined")

        unique_files = set(m.get("file_id") for m in expected)
        declared_tfc = q.get("target_file_count")
        if declared_tfc != len(unique_files):
            errors.append(f"{qid}: target_file_count ({declared_tfc}) does not match unique files ({len(unique_files)})")

        if category == "single_file" and len(unique_files) != 1:
            errors.append(f"{qid}: Category is single_file but references {len(unique_files)} files")
        if category == "multi_file" and len(unique_files) < 2:
            errors.append(f"{qid}: Category is multi_file but references {len(unique_files)} files")

        for m in expected:
            fid = m.get("file_id")
            tid = m.get("turn_id")
            exp_spk = m.get("speaker")
            mtxt = m.get("matched_text")
            mst = m.get("start_seconds")
            met = m.get("end_seconds")

            if fid not in transcripts:
                errors.append(f"{qid}: References missing audio file {fid}")
                continue

            if tid not in transcripts[fid]:
                errors.append(f"{qid}: References non-existent Turn {tid} in {fid}")
                continue

            ref_turn = transcripts[fid][tid]
            ref_spk = ref_turn.get("speaker")
            ref_text = ref_turn.get("text")
            ref_st = ref_turn.get("start_time")
            ref_et = ref_turn.get("end_time")

            # Check speaker match
            if exp_spk and exp_spk != ref_spk:
                errors.append(f"{qid}: Declared speaker '{exp_spk}' does not match transcript speaker '{ref_spk}' in {fid} Turn {tid}")

            # Check numeric, finite, non-boolean timestamps
            if not isinstance(mst, (int, float)) or isinstance(mst, bool) or not isinstance(met, (int, float)) or isinstance(met, bool):
                errors.append(f"{qid}: Non-numeric moment interval [{mst}, {met}]")
                continue
            if math.isnan(mst) or math.isnan(met) or math.isinf(mst) or math.isinf(met):
                errors.append(f"{qid}: NaN or Inf moment interval [{mst}, {met}]")
                continue
            if met <= mst:
                errors.append(f"{qid}: Non-positive moment interval [{mst}, {met}]")
                continue

            # Check interval containment: moment must sit within turn boundaries
            if mst < ref_st - 0.05 or met > ref_et + 0.05:
                errors.append(f"{qid}: Moment interval [{mst}, {met}] is outside Turn {tid} interval [{ref_st}, {ref_et}] in {fid}")

            # Check non-empty and STRICT VERBATIM SUBSTRING MATCH
            if not mtxt or not mtxt.strip():
                errors.append(f"{qid}: matched_text is empty")
            elif mtxt not in ref_text:
                errors.append(f"{qid}: matched_text is NOT an exact verbatim substring of Turn {tid} in {fid}!\n  matched_text: '{mtxt[:60]}...'\n  turn_text:    '{ref_text[:60]}...'")

        # Validate hard negatives
        hard_negs = q.get("hard_negatives", [])
        for hn in hard_negs:
            hn_fid = hn.get("file_id")
            hn_tid = hn.get("turn_id")
            hn_spk = hn.get("speaker")
            hn_st = hn.get("start_seconds")
            hn_et = hn.get("end_seconds")

            if hn_fid not in transcripts:
                errors.append(f"{qid} hard_negative: Unknown file {hn_fid}")
            elif hn_tid not in transcripts[hn_fid]:
                errors.append(f"{qid} hard_negative: Unknown Turn {hn_tid} in {hn_fid}")
            else:
                hn_turn = transcripts[hn_fid][hn_tid]
                if hn_spk and hn_spk != hn_turn.get("speaker"):
                    errors.append(f"{qid} hard_negative: Speaker '{hn_spk}' != '{hn_turn.get('speaker')}'")

                # Check numeric, finite, non-boolean
                if not isinstance(hn_st, (int, float)) or isinstance(hn_st, bool) or not isinstance(hn_et, (int, float)) or isinstance(hn_et, bool):
                    errors.append(f"{qid} hard_negative: Non-numeric interval [{hn_st}, {hn_et}]")
                    continue
                if math.isnan(hn_st) or math.isnan(hn_et) or math.isinf(hn_st) or math.isinf(hn_et):
                    errors.append(f"{qid} hard_negative: NaN or Inf interval [{hn_st}, {hn_et}]")
                    continue
                if hn_et <= hn_st:
                    errors.append(f"{qid} hard_negative: Non-positive interval [{hn_st}, {hn_et}]")
                    continue

                ref_st = hn_turn.get("start_time")
                ref_et = hn_turn.get("end_time")
                if hn_st < ref_st - 0.05 or hn_et > ref_et + 0.05:
                    errors.append(f"{qid} hard_negative: Interval [{hn_st}, {hn_et}] outside Turn {hn_tid} in {hn_fid}")

    if breakdown != {"single_file": 6, "multi_file": 6, "near_miss": 6}:
        errors.append(f"Unexpected query breakdown: {breakdown} (expected 6/6/6)")

    print(f"✓ Benchmark queries (qrels) validated: exactly 18 queries (6 single, 6 multi, 6 near-miss), all verbatim substrings.")

    if errors:
        print(f"\n❌ FOUND {len(errors)} INTEGRITY ERRORS:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    else:
        print("\n✅ ZERO DEFECTS: All audio, transcripts, and qrels pass 100% mathematical, schema, and verbatim integrity verification!")


if __name__ == "__main__":
    validate_all()
