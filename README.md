# GazeTalk Synthetic Conversation Generator

A minimal, research-oriented package for generating profile-conditioned synthetic therapist--patient conversations. It retains only the synthetic conversation pipeline from the larger GazeTalk simulator project.

The four implemented simulation profiles are:

- `anomic`: word-finding and self-repair difficulty;
- `broca`: short, non-fluent, agrammatic-style output;
- `wernicke`: fluent output with possible semantic mismatch or topic drift;
- `global`: very short, high-support, multimodal-style communication.

These are simulation profiles, not clinical diagnoses. The generated data are not a substitute for real patients, clinical assessment, or expert validation.

## Repository scope

Included:

- fictional persona input in JSONL format;
- two labelled 6,000-dialogue synthetic datasets;
- deterministic balanced or profile-quota job planning;
- OpenAI and Anthropic generation backends;
- strict six-turn dialogue validation;
- JSONL cache, retry, error log, and resume support;
- a lightweight structural and language evaluation;
- unit tests that do not call external APIs.

Excluded intentionally:

- the Streamlit platform and static dashboards;
- GazeTalk click/input simulation;
- AphasiaBank transcripts, spreadsheets, and derived participant records;
- the merged 12,000-dialogue derivative and evaluation intermediates;
- API keys, model caches, logs, virtual environments, and report files.

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd gazetalk-synthetic-conversation

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Set one API key in the current shell. Never put a real key in source code:

```bash
export OPENAI_API_KEY="your-key"
```

or:

```bash
export ANTHROPIC_API_KEY="your-key"
```

## Persona format

The public file `examples/personas.jsonl` contains four manually written fictional examples. Each line is one JSON object:

```json
{
  "persona_id": "DEMO_ANOMIC_001",
  "profile": "anomic",
  "age_band": "older adult",
  "occupation_category": "education occupation",
  "severity": "mild",
  "communication_goal": "practice word retrieval in everyday conversation",
  "preferred_support": ["semantic cues", "extra response time"],
  "behaviour_profile": {
    "grammar": "mostly preserved",
    "sentence_length": "medium",
    "word_finding": "high"
  }
}
```

Use only fictional or appropriately governed persona data. Do not publish restricted participant-level records.

## Included datasets

The two original generation batches are included separately under `datasets/`:

| Folder | Sampling design | Records |
|---|---|---:|
| `datasets/balanced_6000/` | Equal allocation across four profiles | 6,000 |
| `datasets/aphasiabank_ratio_6000/` | Allocation based on the selected strictly mapped AphasiaBank subset | 6,000 |

See [`datasets/README.md`](datasets/README.md) for profile counts, provenance, checksums, field descriptions, and limitations. The ratio batch reflects the selected reference subset and is not an estimate of general clinical prevalence.

## Preview the generation plan

`--dry-run` does not call an API and does not require a key:

```bash
gazetalk-generate \
  --personas examples/personas.jsonl \
  --samples-per-persona 2 \
  --seed 42 \
  --dry-run
```

## Generate a small balanced batch

```bash
gazetalk-generate \
  --personas examples/personas.jsonl \
  --samples-per-persona 2 \
  --backend openai \
  --model gpt-4.1-mini \
  --resume \
  --output outputs/dialogues.jsonl
```

For Anthropic, select the backend and explicitly provide a model available to your account:

```bash
gazetalk-generate \
  --personas examples/personas.jsonl \
  --samples-per-persona 2 \
  --backend anthropic \
  --model YOUR_ANTHROPIC_MODEL \
  --resume \
  --output outputs/dialogues.jsonl
```

## Generate exact profile quotas

The quota mode supports a prescribed sampling distribution. For example, the earlier 6,000-dialogue AphasiaBank-ratio experiment used:

```bash
gazetalk-generate \
  --personas /path/to/private_personas.jsonl \
  --profile-counts anomic=2628,broca=2529,wernicke=521,global=322 \
  --id-prefix R2_ \
  --backend openai \
  --model gpt-4.1-mini \
  --seed 42 \
  --resume \
  --output outputs/dialogues_ratio.jsonl
```

The counts reproduce a selected reference subset; they must not be interpreted as general clinical prevalence.

## Resume and failure handling

Each successful record is flushed immediately. If a run stops, repeat the same command with `--resume`. Existing `dialogue_id` values are skipped. Records that still fail after all attempts are written to:

```text
OUTPUT_FILE.errors.jsonl
```

Prompt-response caching defaults to `outputs/llm_cache.jsonl`. The entire `outputs/` directory is ignored by Git.

## Output format

The output stays compatible with the earlier generated corpus:

```json
{
  "persona_id": "DEMO_ANOMIC_001",
  "dialogue_id": "DEMO_ANOMIC_001_D1",
  "profile": "anomic",
  "session_index": 1,
  "session_topic": "daily routine",
  "therapist_strategy": "semantic cueing",
  "backend": "openai",
  "model": "gpt-4.1-mini",
  "cached": false,
  "input_tokens": 300,
  "output_tokens": 180,
  "attempt": 1,
  "latency_local": 3.2,
  "text": "{\"dialogue\": [{\"speaker\": \"therapist\", \"utterance\": \"...\"}]}"
}
```

The nested `text` value is a JSON string with exactly six alternating therapist/patient turns.

## Basic evaluation

```bash
gazetalk-evaluate \
  --input outputs/dialogues.jsonl \
  --output-dir outputs/evaluation
```

The evaluator reports parse success, profile counts, exact patient-text and complete-dialogue duplication, patient words, MLU, TTR, corrected within-synthetic LIX, textual ellipsis markers, and brief yes/no-led turns. These are descriptive or surface-based measures, not clinical annotations.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Before publishing to GitHub

```bash
git status --short
git check-ignore -v outputs/llm_cache.jsonl
rg -n --hidden --glob '!.git/**' 'sk-(proj-)?[A-Za-z0-9_-]{20,}'
rg -n --hidden --glob '!.git/**' --glob '!.env.example' \
  '^(OPENAI_API_KEY|ANTHROPIC_API_KEY)=.+'
```

The last command should not reveal any real credential. Review every staged file before committing.

## Research and data statement

This repository contains code, fictional example personas, and model-generated synthetic conversations. It does not include AphasiaBank transcripts, original participant identifiers, spreadsheets, audio, or video. AphasiaBank materials are access-controlled and are not redistributed here. Any use of private corpora must follow the applicable data-use agreement and institutional requirements.

## License

MIT. See `LICENSE`.
