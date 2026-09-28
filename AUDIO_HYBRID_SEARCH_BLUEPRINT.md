# Conversational Audio Hybrid Search: Architecture, Dataset & Evaluation Blueprint

## 1. Executive Summary & Problem Framing

The objective is to design, implement, and evaluate a production-grade, local hybrid search engine (Lexical + Semantic) over conversational audio recordings. Each recording consists of an 8–10 minute two-speaker dialogue across diverse topic domains and acoustic environments.

The system addresses the core challenge of audio retrieval: **how to allow users to search for both verbatim spoken words and abstract concepts, returning results that pinpoint the exact audio file, millisecond-accurate timestamp, and speaker identity.**

To ensure modularity and debuggability, the problem is decoupled into two independent phases:
- **Phase 1: Conversion Pipeline** (Audio $\rightarrow$ Word-level ASR $\rightarrow$ Speaker Diarization $\rightarrow$ Turn-by-Turn Structured Transcript).
- **Phase 2: Storage, Retrieval & Fusion Pipeline** (Semantic Utterance Chunking $\rightarrow$ Dual Indexing in PostgreSQL/pgvector $\rightarrow$ Reciprocal Rank Fusion $\rightarrow$ Evaluation via Recall@k).

---

## 2. Overall System Architecture

```
                                 ┌───────────────────────────────────────────────┐
                                 │   Input Audio (5–6 files, 8–10 minutes each)  │
                                 │        Diverse Domains & Acoustic SNRs        │
                                 └───────────────────────┬───────────────────────┘
                                                         │
═════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════
PHASE 1: CONVERSION & DIARIZATION PIPELINE               │
═════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════
                                                         ▼
                                 ┌───────────────────────────────────────────────┐
                                 │ 1. Ingestion & Preprocessing (ffmpeg)         │
                                 │    - 16 kHz, 16-bit Mono PCM WAV             │
                                 │    - Loudness normalization (-23 LUFS / Peak) │
                                 │    - Silero-VAD (Speech boundary gating)      │
                                 └───────────────────────┬───────────────────────┘
                                                         │
                                ┌────────────────────────┴────────────────────────┐
                                │                                                 │
                                ▼                                                 ▼
             ┌─────────────────────────────────────┐           ┌─────────────────────────────────────┐
             │ 2a. ASR Transcription               │           │ 2b. Speaker Diarization             │
             │     - faster-whisper / distil-v3    │           │     - PyAnnote.audio 3.1            │
             │     - Token stream output           │           │     - Constraint: k=2 speakers      │
             └──────────────────┬──────────────────┘           │     - Outputs temporal intervals    │
                                │                              └──────────────────┬──────────────────┘
                                ▼                                                 │
             ┌─────────────────────────────────────┐                              │
             │ 2c. Forced Alignment (Wav2Vec2)     │                              │
             │     - Phoneme-to-audio frame sync   │                              │
             │     - Millisecond word timestamps   │                              │
             └──────────────────┬──────────────────┘                              │
                                │                                                 │
                                └────────────────────────┬────────────────────────┘
                                                         │
                                                         ▼
                                 ┌───────────────────────────────────────────────┐
                                 │ 3. Alignment & Speaker Turn Reconciliation     │
                                 │    - Map word intervals to speaker frames     │
                                 │    - Resolve edge overlaps & crosstalk        │
                                 │    - Output: Turn-by-Turn Canonical Transcript│
                                 └───────────────────────┬───────────────────────┘
                                                         │
═════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════
PHASE 2: INDEXING, HYBRID SEARCH & RETRIEVAL             │
═════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════
                                                         ▼
                                 ┌───────────────────────────────────────────────┐
                                 │ 4. Context-Aware Semantic Chunking            │
                                 │    - Aggregate contiguous turns / QA pairs    │
                                 │    - Window: 60–90 seconds (150–250 words)    │
                                 │    - 15–20s sliding window overlap            │
                                 └───────────────────────┬───────────────────────┘
                                                         │
                                ┌────────────────────────┴────────────────────────┐
                                │                                                 │
                                ▼                                                 ▼
             ┌─────────────────────────────────────┐           ┌─────────────────────────────────────┐
             │ 5a. Lexical Tokenization            │           │ 5b. Local Dense Embedding           │
             │     - English dictionary stemming   │           │     - all-MiniLM-L6-v2 / BGE-small  │
             │     - PostgreSQL tsvector           │           │     - 384-dimensional dense vectors │
             └──────────────────┬──────────────────┘           └──────────────────┬──────────────────┘
                                │                                                 │
                                └────────────────────────┬────────────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            PostgreSQL 16 + pgvector                                             │
│                                                                                                                 │
│   audio_segments (id, audio_file_id, speaker_label, start_time, end_time, time_range, content, embedding)       │
│                                                                                                                 │
│             ┌────────────────────────────────────┐       ┌────────────────────────────────────┐                 │
│             │ GIN Index (content_tsv)            │       │ HNSW Index (vector_cosine_ops)     │                 │
│             │ Keyword / Lexical exact matching   │       │ Semantic similarity (m=16, ef=128) │                 │
│             └─────────────────┬──────────────────┘       └─────────────────┬──────────────────┘                 │
└───────────────────────────────┼─────────────────────────────────────────────┼───────────────────────────────────┘
                                │                                             │
                                ▼                                             ▼
                          Lexical Rank                                   Dense Rank
                                │                                             │
                                └──────────────────────┬──────────────────────┘
                                                       │
                                                       ▼
                                ┌───────────────────────────────────────────────┐
                                │ 6. Reciprocal Rank Fusion (RRF)               │
                                │    Score(d) = Σ 1 / (k + rank_i(d)), k = 60   │
                                └──────────────────────┬───────────────────────┘
                                                       │
                                                       ▼
                                ┌───────────────────────────────────────────────┐
                                │ 7. Search Output & Timestamp Pinpointing      │
                                │    - File, Speaker, Time range [start - end]  │
                                │    - Jump-to-second audio offset              │
                                │    - Highlighted conversational snippet       │
                                └───────────────────────────────────────────────┘
```

---

## 3. Phase 1: Conversion Pipeline In-Depth

### Step 1: Preprocessing & Normalization
- **Format:** Audio decoded and converted to `16,000 Hz`, single-channel `mono`, `16-bit PCM WAV`.
- **Loudness Normalization:** Standardized to `-23 LUFS` (or peak `-1.0 dBFS`) using `ffmpeg` filters (`loudnorm`) to avoid dropping quiet speakers.
- **Voice Activity Detection (VAD):** Deep-learning VAD (Silero) strips leading/trailing silences and filters background murmur.

### Step 2: ASR (Automatic Speech Recognition) + Forced Alignment
- **Transcription Engine:** `faster-whisper` (CTranslate2 backend) running `distil-whisper/distil-large-v3` or `large-v3`.
- **Forced Alignment (Wav2Vec2):** Whisper's native timestamps drift across long windows. Forced phoneme alignment anchors every single word to absolute millisecond boundaries (`start`, `end`, `confidence`).

### Step 3: Diarization & Constrained Clustering
- **Diarization Engine:** `pyannote.audio 3.1`.
- **Constraint Enforcement:** `min_speakers=2, max_speakers=2`. In noisy or conversational clips, unconstrained clustering produces phantom speakers (e.g. `SPEAKER_02`). Hardcoding $k=2$ guarantees speaker consistency.

### Step 4: Word-to-Speaker Assignment (Reconciliation)
- Temporal intersection maps each aligned word $[t_{start}, t_{end}]$ to the active speaker interval with maximal overlap.
- Words falling into brief pauses ($< 200\text{ms}$) inherit the surrounding turn's speaker.
- **Phase 1 Output Contract:** Canonical Turn-by-Turn JSON.

#### Example Turn-by-Turn Output:
```json
[
  {
    "turn_id": 14,
    "speaker": "SPEAKER_00",
    "start_time": 74.10,
    "end_time": 76.10,
    "text": "Have you considered pgvector for this?",
    "words": [
      { "word": "Have", "start": 74.10, "end": 74.30 },
      { "word": "you", "start": 74.32, "end": 74.45 },
      { "word": "considered", "start": 74.48, "end": 74.95 },
      { "word": "pgvector", "start": 75.02, "end": 75.60 },
      { "word": "for", "start": 75.62, "end": 75.75 },
      { "word": "this?", "start": 75.78, "end": 76.10 }
    ]
  },
  {
    "turn_id": 15,
    "speaker": "SPEAKER_01",
    "start_time": 76.80,
    "end_time": 81.60,
    "text": "Yes, we benchmarked pgvector against Pinecone. We saw latency drop significantly.",
    "words": [
      { "word": "Yes,", "start": 76.80, "end": 77.10 },
      { "word": "we", "start": 77.20, "end": 77.35 },
      { "word": "benchmarked", "start": 77.38, "end": 77.90 },
      { "word": "pgvector", "start": 77.95, "end": 78.45 },
      { "word": "against", "start": 78.50, "end": 78.80 },
      { "word": "Pinecone.", "start": 78.85, "end": 79.40 },
      { "word": "We", "start": 79.70, "end": 79.85 },
      { "word": "saw", "start": 79.88, "end": 80.05 },
      { "word": "latency", "start": 80.10, "end": 80.55 },
      { "word": "drop", "start": 80.60, "end": 80.85 },
      { "word": "significantly.", "start": 80.90, "end": 81.60 }
    ]
  }
]
```

---

## 4. Phase 2: Indexing, Hybrid Search & RRF

### Conversational Semantic Chunking
- **Why naive chunking fails:** Fixed-second chunking (e.g. 30s) cuts sentences in half; single-turn chunking isolates short replies ("Yes", "Exactly") from their context.
- **Utterance Merging Strategy:** Merge adjacent speaker turns into a coherent dialogue block of **60–90 seconds** (150–250 words) with a **15–20 second sliding window**.
- Boundaries strictly snap to speaker turn endings or sentence punctuation.

### Dual Indexing in PostgreSQL
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE audio_segments (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    speaker_label VARCHAR(50) NOT NULL,
    start_time NUMERIC(8, 2) NOT NULL,
    end_time NUMERIC(8, 2) NOT NULL,
    time_range numrange GENERATED ALWAYS AS (numrange(start_time, end_time)) STORED,
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(384) NOT NULL
);

-- Lexical Inverted Index
CREATE INDEX idx_segments_tsv ON audio_segments USING GIN (content_tsv);

-- Dense Vector Graph Index
CREATE INDEX idx_segments_hnsw ON audio_segments USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Temporal Range Index
CREATE INDEX idx_segments_time ON audio_segments USING GIST (time_range);
```

### Fusion Algorithm: Reciprocal Rank Fusion (RRF)
BM25/FTS scores are unbounded $[0, \infty)$, while cosine similarities sit in $[-1, 1]$. To avoid fragile score-normalization tuning, we use RRF:

$$RRF\_Score(d) = \sum_{m \in \{lexical, dense\}} \frac{1}{60 + \text{rank}_m(d)}$$

---

## 5. Golden Dataset Specification (7 Audio Files)

To guarantee that the benchmark reflects real-world longevity rather than overfitting to clean tech interviews, the dataset spans **seven diverse topics, distinct speaker pairs, and varying acoustic quality profiles**:

| File ID | Domain / Topic | Speakers (2 unique) | Acoustic Quality & Noise Profile | What This Stress-Tests |
| :--- | :--- | :--- | :--- | :--- |
| **`audio_01.wav`** | **Software Architecture** (Systems & OS) | Lex Fridman + DHH | **Studio Clean (High SNR):** Shure SM7B, acoustically treated. | Baseline retrieval. Systems jargon (`Omarchy`, `Hyprland`, `Wayland`, `C++`). |
| **`audio_02.wav`** | **Database Internals & Vector Search** | Michael Christofides + Jonathan S. Katz | **Remote VoIP:** Broadcast internet connection across London & NYC. | Database internals jargon (`pgvector`, `HNSW`, `IVFFlat`, `GiST`, `8KB page limit`). |
| **`audio_03.wav`** | **Psychology & Cognitive Neuroscience** | Adam Stacoviak + Dr. Mireille B. Reece | **Remote Studio Call:** Clean broadcast VoIP. | Clinical psychology terminology (attachment theory, emotional regulation, ADHD, metaphors). |
| **`audio_04.wav`** | **Macroeconomics & Capital Markets** | Dwarkesh Patel + Dylan Patel | **Studio Broadcast:** High SNR studio conversation. | Hardware macroeconomics (`Fab CapEx`, `$6 billion per gigawatt`, `N3/N5 wafers`, `ASML`). |
| **`audio_05.wav`** | **Sports Tactics & Combat Debate** | Lex Fridman + Khabib Nurmagomedov | **Studio Broadcast:** High dynamic range conversation. | Combat sports tactics (`top pressure`, `wrist control`, `leg triangle`, `Conor McGregor fight strategy`). |
| **`audio_06.wav`** | **Constitutional Jurisprudence & Law** | Tyler Cowen + Cass Sunstein | **Faculty Office Studio:** Natural room acoustics. | Legal philosophy and jurisprudence (`Kantian liberalism`, `rule of law`, `dignity under state coercion`). |
| **`audio_07.wav`** | **Investigative Military History** | Lex Fridman + Gary Gallagher | **Authentic Cafe Chatter:** Mixed at 12.0 dB SNR from DEMAND dataset (CC BY 4.0). | Background noise robustness, VAD boundary gating, Whisper anti-hallucination, and diarization. |

### Dataset File Specifications:
- **Duration:** 8:00 to 10:00 minutes per clip (506.0s to 584.0s).
- **Format:** 16kHz, 1-channel Mono, 16-bit PCM WAV.
- **Speaker Invariant:** Exactly 2 active foreground interlocutors per audio file.

---

## 6. Evaluation Framework & Recall@k Definition

### What is Recall@k in Conversational Audio Retrieval?
Recall@k measures whether the search system successfully returned the correct audio moment within the top $k$ retrieved results.

Because audio has a continuous temporal dimension, a retrieved result $r$ is defined as a **True Positive** if and only if:
1. `file_id` matches the ground truth file.
2. `speaker` matches the expected speaker.
3. **Strict Temporal Match:** strictly positive overlap (`overlap > 0`) AND (`Temporal IoU >= 0.3` OR `|start_pred - start_gt| <= 5.0s` with minimum 25% coverage).

$$\text{Recall@k} = \frac{\text{Number of Ground Truth moments retrieved in top } k}{\text{Total Ground Truth moments in query set}}$$

### Golden Labeled Query Benchmark (18 queries in `dataset/qrels/benchmark_queries.json`):
1. **Single-File Queries (6 queries):** Needle-in-a-haystack retrieval isolated from 6 distractor files.
2. **Multi-File Queries (6 queries):** Conceptual themes spanning 2-3 files to evaluate cross-corpus recall.
3. **Near-Miss Queries (6 queries):** Hard semantic distractors that evaluate retrieval precision and reject false positives at Rank #1.

### Success Benchmark Criteria (Automated Promptfoo & Recall Suite):
- **Hybrid RRF Recall@5 $\ge 0.85$**
- **Hybrid RRF Recall@1 $\ge 0.60$**
- **Strict Ablation Proof:** Hybrid RRF Recall@5 must strictly outperform both Lexical-only and Dense-only baselines (`Hybrid > Lexical` AND `Hybrid > Dense`).
