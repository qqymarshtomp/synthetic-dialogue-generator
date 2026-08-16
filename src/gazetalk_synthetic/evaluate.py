from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .schema import validate_dialogue_text


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…+)")
SENTENCE_END_RE = re.compile(r"[.!?]+")


def normalize_text(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text).lower())


def parse_record(item: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    raw = item.get("text", "")
    if isinstance(raw, dict):
        text = json.dumps(raw, ensure_ascii=False)
    elif isinstance(raw, str):
        text = raw.strip()
    else:
        raise ValueError("text must be a JSON string or object.")
    parsed = validate_dialogue_text(text)
    dialogue = parsed["dialogue"]
    patient = [turn["utterance"].strip() for turn in dialogue if turn["speaker"] == "patient"]
    return patient, dialogue


def calculate_lix(utterances: list[str], words: list[str]) -> float:
    if not words:
        return 0.0
    sentence_count = 0
    for utterance in utterances:
        without_ellipses = ELLIPSIS_RE.sub(" ", normalize_text(utterance))
        parts = [part for part in SENTENCE_END_RE.split(without_ellipses) if part.strip()]
        sentence_count += max(1, len(parts))
    long_words = sum(len(word) > 6 for word in words)
    return len(words) / sentence_count + 100.0 * long_words / len(words)


def normalize_for_duplicate(text: str) -> str:
    return " ".join(tokenize(text))


def mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated dialogue JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    profile_totals: Counter[str] = Counter()
    total_lines = 0
    patient_duplicates: Counter[str] = Counter()
    full_duplicates: Counter[str] = Counter()
    persona_texts: dict[str, Counter[str]] = defaultdict(Counter)

    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                item = json.loads(line)
                profile = str(item.get("profile", "unknown")).lower()
                profile_totals[profile] += 1
                patient_utterances, dialogue = parse_record(item)
                patient_text = " ".join(patient_utterances)
                words = tokenize(patient_text)
                patient_turns = len(patient_utterances)
                ellipses = len(ELLIPSIS_RE.findall(patient_text))
                brief_yes_no = sum(
                    len(turn_words) <= 3 and turn_words and turn_words[0] in {"yes", "no"}
                    for turn_words in (tokenize(value) for value in patient_utterances)
                )
                patient_key = normalize_for_duplicate(patient_text)
                full_key = normalize_for_duplicate(" ".join(
                    f"{turn['speaker']} {turn['utterance']}" for turn in dialogue
                ))
                patient_duplicates[patient_key] += 1
                full_duplicates[full_key] += 1
                persona_texts[str(item.get("persona_id", ""))][patient_key] += 1
                rows.append({
                    "line_number": line_number,
                    "persona_id": item.get("persona_id", ""),
                    "dialogue_id": item.get("dialogue_id", ""),
                    "profile": profile,
                    "patient_turns": patient_turns,
                    "patient_words": len(words),
                    "mean_utterance_length": len(words) / patient_turns,
                    "ttr": len(set(words)) / len(words) if words else 0.0,
                    "lix": calculate_lix(patient_utterances, words),
                    "ellipsis_markers_per_100_words": 100.0 * ellipses / len(words) if words else 0.0,
                    "brief_yes_no_turn_ratio": brief_yes_no / patient_turns,
                })
            except Exception as exc:
                invalid.append({"line_number": line_number, "error": str(exc)})

    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row)

    profile_summary = []
    for profile in sorted(profile_totals):
        group = by_profile[profile]
        profile_summary.append({
            "profile": profile,
            "total_records": profile_totals[profile],
            "valid_records": len(group),
            "parse_success_rate": len(group) / profile_totals[profile],
            "mean_patient_words_per_dialogue": mean([row["patient_words"] for row in group]),
            "mean_utterance_length": mean([row["mean_utterance_length"] for row in group]),
            "mean_ttr": mean([row["ttr"] for row in group]),
            "mean_lix": mean([row["lix"] for row in group]),
            "mean_ellipsis_markers_per_100_words": mean([row["ellipsis_markers_per_100_words"] for row in group]),
            "mean_brief_yes_no_turn_ratio": mean([row["brief_yes_no_turn_ratio"] for row in group]),
        })

    total = total_lines
    valid = len(rows)
    patient_extras = sum(max(0, count - 1) for count in patient_duplicates.values())
    full_extras = sum(max(0, count - 1) for count in full_duplicates.values())
    same_persona_extras = sum(
        max(0, count - 1)
        for texts in persona_texts.values()
        for count in texts.values()
    )
    summary = {
        "input_file": args.input,
        "total_records": total,
        "valid_records": valid,
        "invalid_records": len(invalid),
        "parse_success_rate": valid / total if total else 0.0,
        "profile_counts": dict(profile_totals),
        "patient_text_duplicate_samples": patient_extras,
        "patient_text_duplicate_rate": patient_extras / valid if valid else 0.0,
        "same_persona_duplicate_samples": same_persona_extras,
        "same_persona_duplicate_rate": same_persona_extras / valid if valid else 0.0,
        "complete_dialogue_duplicate_samples": full_extras,
        "complete_dialogue_duplicate_rate": full_extras / valid if valid else 0.0,
        "notes": {
            "lix": "Descriptive within-corpus measure using ellipsis-normalized sentence segmentation.",
            "ellipsis": "Textual marker, not an acoustic pause measure.",
            "brief_yes_no": "Brief yes/no-led turn ratio, not direct prompt dependency.",
        },
    }

    with (output_dir / "record_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["line_number"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "profile_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(profile_summary[0]) if profile_summary else ["profile"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profile_summary)
    with (output_dir / "invalid_records.jsonl").open("w", encoding="utf-8") as handle:
        for item in invalid:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
