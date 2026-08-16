# Included synthetic conversation datasets

This directory contains two independently generated 6,000-dialogue batches. They are stored separately so that their sampling designs remain explicit.

## Dataset inventory

| Dataset | File | Anomic | Broca | Wernicke | Global | Total |
|---|---|---:|---:|---:|---:|---:|
| Balanced | `balanced_6000/dialogues.jsonl` | 1,500 | 1,500 | 1,500 | 1,500 | 6,000 |
| AphasiaBank-ratio | `aphasiabank_ratio_6000/dialogues.jsonl` | 2,628 | 2,529 | 521 | 322 | 6,000 |

Both files contain:

- 6,000 parseable JSONL records;
- 6,000 unique `dialogue_id` values;
- 200 synthetic persona identifiers;
- exactly six alternating therapist/patient turns per dialogue;
- 15 session-topic categories and eight therapist-strategy categories;
- generation metadata including backend, model, token counts, cache status, and local latency.

Basic validation with the included evaluator produced:

| Dataset | Parse success | Repeated patient texts beyond first | Same-persona repeats beyond first | Repeated complete dialogues beyond first |
|---|---:|---:|---:|---:|
| Balanced | 100% | 103 (1.72%) | 5 (0.08%) | 4 (0.07%) |
| AphasiaBank-ratio | 100% | 21 (0.35%) | 1 (0.02%) | 0 (0.00%) |

Patient-text repetition is calculated using only normalized patient utterances. Short responses can recur while the surrounding therapist--patient dialogue remains different, so it should not be interpreted as complete-dialogue duplication.

All 12,000 records were generated with the OpenAI backend and the recorded model `gpt-4.1-mini`. Model and token metadata are retained in each row for reproducibility.

## Balanced batch

The balanced batch assigns 1,500 dialogues to each of the four simulation profiles. It is intended for controlled profile comparisons and experiments where equal profile exposure is useful.

## AphasiaBank-ratio batch

The second batch uses exact quotas derived by normalizing the strictly mapped Time-1 Boston-type counts across the four implemented profiles:

```text
Anomic:    106 / 242 -> 2,628 dialogues
Broca:     102 / 242 -> 2,529 dialogues
Wernicke:   21 / 242 ->   521 dialogues
Global:     13 / 242 ->   322 dialogues
```

This distribution describes the selected mapped AphasiaBank subset. It must not be interpreted as the general clinical prevalence of aphasia types.

## Record format

Each line is one outer JSON object. The `text` field is itself a JSON string containing the six dialogue turns.

```json
{
  "persona_id": "SYN_ANOMIC_00001",
  "dialogue_id": "SYN_ANOMIC_00001_D1",
  "profile": "anomic",
  "session_index": 1,
  "session_topic": "daily routine",
  "therapist_strategy": "semantic cueing",
  "backend": "openai",
  "model": "gpt-4.1-mini",
  "cached": false,
  "input_tokens": 380,
  "output_tokens": 180,
  "attempt": 1,
  "latency_local": 3.5,
  "text": "{\"dialogue\": [{\"speaker\": \"therapist\", \"utterance\": \"...\"}, {\"speaker\": \"patient\", \"utterance\": \"...\"}]}"
}
```

## Provenance and privacy

The conversations are model-generated synthetic text. Persona identifiers beginning with `SYN_` or `R2_SYN_` are synthetic identifiers, not AphasiaBank participant identifiers. No source transcript, original participant ID, spreadsheet, audio, or video is included.

The persona design and second-batch allocation were informed by aggregate or derived analysis of access-controlled research data. Users remain responsible for complying with the relevant data-use agreement when reproducing that derivation from private source data.

## Intended use

Appropriate uses include generation-pipeline evaluation, exploratory NLP experiments, profile-classification baselines, and research on synthetic conversation methodology.

The datasets are not clinically validated and must not be used for diagnosis, treatment decisions, direct patient assessment, or claims that the simulated profiles fully reproduce real aphasic communication.

## Checksums

```text
24f9cd2a3c8dead5a8e66a48ab42976e95d37823a9a1cd7df47296d6eb6892b6  balanced_6000/dialogues.jsonl
ba9372e23a905a69a09abdd6ca8e533368ec2a6426cb3d4787946b2d97e65f74  aphasiabank_ratio_6000/dialogues.jsonl
```

Verify from this directory with:

```bash
shasum -a 256 balanced_6000/dialogues.jsonl aphasiabank_ratio_6000/dialogues.jsonl
```

## Licensing note

The repository's MIT license covers the software. A separate dataset license has not been asserted in this package; the repository owner should choose and document one before public data redistribution.
