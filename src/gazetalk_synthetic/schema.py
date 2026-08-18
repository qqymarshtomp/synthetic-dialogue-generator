from __future__ import annotations

import json
from typing import Any


EXPECTED_SPEAKERS = (
    "therapist",
    "patient",
    "therapist",
    "patient",
    "therapist",
    "patient",
)

OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "aphasia_dialogue",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "dialogue": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "enum": ["therapist", "patient"],
                            },
                            "utterance": {"type": "string", "minLength": 1},
                        },
                        "required": ["speaker", "utterance"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["dialogue"],
            "additionalProperties": False,
        },
    },
}


def validate_dialogue_text(text: str) -> dict[str, Any]:
    """Parse and validate the six-turn dialogue contract."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object.")

    dialogue = parsed.get("dialogue")
    if not isinstance(dialogue, list):
        raise ValueError("Response must contain a dialogue list.")
    if len(dialogue) != len(EXPECTED_SPEAKERS):
        raise ValueError(f"Expected 6 turns, received {len(dialogue)}.")

    for index, expected_speaker in enumerate(EXPECTED_SPEAKERS):
        turn = dialogue[index]
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {index + 1} must be an object.")
        if set(turn) != {"speaker", "utterance"}:
            raise ValueError(
                f"Turn {index + 1} must contain only speaker and utterance."
            )
        if turn["speaker"] != expected_speaker:
            raise ValueError(
                f"Turn {index + 1} has speaker={turn['speaker']!r}; "
                f"expected {expected_speaker!r}."
            )
        utterance = turn["utterance"]
        if not isinstance(utterance, str) or not utterance.strip():
            raise ValueError(f"Turn {index + 1} has an empty utterance.")

    return parsed

