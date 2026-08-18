from __future__ import annotations

import json
import unittest

from gazetalk_synthetic.evaluate import calculate_lix, tokenize
from gazetalk_synthetic.generate import parse_profile_counts, plan_jobs
from gazetalk_synthetic.schema import validate_dialogue_text


PERSONAS = [
    {"persona_id": "A1", "profile": "anomic"},
    {"persona_id": "B1", "profile": "broca"},
    {"persona_id": "W1", "profile": "wernicke"},
    {"persona_id": "G1", "profile": "global"},
]


def valid_dialogue() -> str:
    dialogue = []
    for index in range(3):
        dialogue.extend([
            {"speaker": "therapist", "utterance": f"Question {index + 1}?"},
            {"speaker": "patient", "utterance": f"Answer {index + 1}."},
        ])
    return json.dumps({"dialogue": dialogue})


class CoreTests(unittest.TestCase):
    def test_validate_dialogue_contract(self) -> None:
        parsed = validate_dialogue_text(valid_dialogue())
        self.assertEqual(len(parsed["dialogue"]), 6)

    def test_validate_rejects_wrong_speaker_order(self) -> None:
        parsed = json.loads(valid_dialogue())
        parsed["dialogue"][0]["speaker"] = "patient"
        with self.assertRaises(ValueError):
            validate_dialogue_text(json.dumps(parsed))

    def test_balanced_job_planning_is_deterministic(self) -> None:
        first = plan_jobs(PERSONAS, 2, None, seed=42)
        second = plan_jobs(PERSONAS, 2, None, seed=42)
        self.assertEqual(
            [item["dialogue_id"] for item in first],
            [item["dialogue_id"] for item in second],
        )
        self.assertEqual(len(first), 8)
        self.assertEqual(len({item["dialogue_id"] for item in first}), 8)

    def test_exact_profile_quotas(self) -> None:
        quotas = parse_profile_counts("anomic=3,broca=2,wernicke=1,global=4")
        jobs = plan_jobs(PERSONAS, 1, quotas, seed=7, id_prefix="R2_")
        counts = {
            profile: sum(item["profile"] == profile for item in jobs)
            for profile in quotas
        }
        self.assertEqual(counts, quotas)
        self.assertEqual(len({item["dialogue_id"] for item in jobs}), 10)
        self.assertTrue(
            all(item["dialogue_id"].startswith("R2_") for item in jobs)
        )

    def test_lix_does_not_treat_ellipsis_as_sentence_boundary(self) -> None:
        utterances = ["I... need the medicine. Yes."]
        words = tokenize(utterances[0])
        self.assertEqual(len(words), 5)
        self.assertAlmostEqual(calculate_lix(utterances, words), 22.5)


if __name__ == "__main__":
    unittest.main()
