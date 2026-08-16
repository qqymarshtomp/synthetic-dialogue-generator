from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .clients import LLMClient, LLMResponse
from .schema import validate_dialogue_text


PROFILES = ("anomic", "broca", "wernicke", "global")

TOPICS = (
    "daily routine",
    "family memories",
    "favorite hobbies",
    "health and wellbeing",
    "previous work experience",
    "weekend activities",
    "food and cooking",
    "travel experiences",
    "community activities",
    "personal stories",
    "shopping and errands",
    "a recent appointment",
    "weather and seasonal activities",
    "home tasks",
    "friends and social events",
)

THERAPIST_STRATEGIES = (
    "open questions followed by one focused prompt",
    "yes-no confirmation when support is needed",
    "semantic cueing",
    "phonological cueing",
    "extra response time with minimal interruption",
    "written keyword support",
    "choice between two options",
    "gesture and pointing support",
)


def load_personas(path: str | Path) -> list[dict[str, Any]]:
    personas: list[dict[str, Any]] = []
    seen: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at persona line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"Persona line {line_number} is not an object.")

            persona_id = str(item.get("persona_id", "")).strip()
            profile = str(item.get("profile", "")).strip().lower()
            if not persona_id:
                raise ValueError(f"Persona line {line_number} has no persona_id.")
            if persona_id in seen:
                raise ValueError(f"Duplicate persona_id: {persona_id}")
            if profile not in PROFILES:
                raise ValueError(
                    f"Persona {persona_id} has unsupported profile {profile!r}."
                )

            item["persona_id"] = persona_id
            item["profile"] = profile
            personas.append(item)
            seen.add(persona_id)

    if not personas:
        raise ValueError("No personas were loaded.")
    return personas


def parse_profile_counts(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    counts: dict[str, int] = {}
    for part in value.split(","):
        name, separator, raw_count = part.strip().partition("=")
        profile = name.strip().lower()
        if not separator or profile not in PROFILES:
            raise ValueError(f"Invalid profile quota: {part!r}")
        count = int(raw_count)
        if count < 0:
            raise ValueError("Profile counts cannot be negative.")
        counts[profile] = count
    return counts


def _job_from_persona(
    persona: dict[str, Any],
    session_index: int,
    seed: int,
    id_prefix: str,
) -> dict[str, Any]:
    job = dict(persona)
    job["session_index"] = session_index
    job["dialogue_id"] = (
        f"{id_prefix}{persona['persona_id']}_D{session_index}"
    )
    rng = random.Random(f"{seed}:{job['dialogue_id']}")
    persona_topics = [
        str(topic).strip()
        for topic in persona.get("topics", [])
        if str(topic).strip()
    ]
    job["session_topic"] = rng.choice(persona_topics or list(TOPICS))
    job["therapist_strategy"] = rng.choice(list(THERAPIST_STRATEGIES))
    return job


def plan_jobs(
    personas: list[dict[str, Any]],
    samples_per_persona: int,
    profile_counts: dict[str, int] | None,
    seed: int,
    id_prefix: str = "",
) -> list[dict[str, Any]]:
    if samples_per_persona < 1:
        raise ValueError("samples_per_persona must be at least 1.")

    jobs: list[dict[str, Any]] = []
    if profile_counts is None:
        for persona in personas:
            for session_index in range(1, samples_per_persona + 1):
                jobs.append(
                    _job_from_persona(persona, session_index, seed, id_prefix)
                )
    else:
        by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for persona in personas:
            by_profile[persona["profile"]].append(persona)

        for profile in PROFILES:
            quota = profile_counts.get(profile, 0)
            candidates = list(by_profile.get(profile, []))
            if quota and not candidates:
                raise ValueError(f"No persona is available for profile {profile!r}.")
            random.Random(f"{seed}:{profile}").shuffle(candidates)
            session_counts: dict[str, int] = defaultdict(int)
            for index in range(quota):
                persona = candidates[index % len(candidates)]
                persona_id = persona["persona_id"]
                session_counts[persona_id] += 1
                jobs.append(
                    _job_from_persona(
                        persona,
                        session_counts[persona_id],
                        seed,
                        id_prefix,
                    )
                )

    random.Random(seed).shuffle(jobs)
    return jobs


def build_prompt(persona: dict[str, Any]) -> tuple[str, str]:
    behaviour = (
        persona.get("behaviour_profile")
        or persona.get("behavior_profile")
        or {}
    )
    support = persona.get("preferred_support") or []

    system_prompt = """
You generate fully synthetic therapist-patient conversations for AAC and
aphasia rehabilitation research. The patient is fictional. Do not copy or
reconstruct any source transcript, real participant, or identifiable case.

Produce exactly six turns, beginning with the therapist and alternating
therapist, patient, therapist, patient, therapist, patient. Every turn must
contain exactly two fields: speaker and utterance. The speaker must be exactly
therapist or patient. Return only one JSON object with a dialogue array.
""".strip()

    user_prompt = f"""
Synthetic persona:
{json.dumps({
    "persona_id": persona.get("persona_id"),
    "profile": persona.get("profile"),
    "age_band": persona.get("age_band"),
    "occupation_category": persona.get("occupation_category"),
    "severity": persona.get("severity"),
    "communication_goal": persona.get("communication_goal"),
    "preferred_support": support,
    "behaviour_profile": behaviour,
    "fatigue_sensitivity": persona.get("fatigue_sensitivity"),
}, ensure_ascii=False, indent=2)}

Session:
{json.dumps({
    "dialogue_id": persona.get("dialogue_id"),
    "session_index": persona.get("session_index"),
    "topic": persona.get("session_topic"),
    "therapist_strategy": persona.get("therapist_strategy"),
}, ensure_ascii=False, indent=2)}

Requirements:
1. Generate exactly three therapist-patient exchanges and six turns.
2. Reflect the profile without placing every possible symptom in every turn.
3. Keep therapist language supportive, concise, and non-diagnostic.
4. Make the content specific to this session topic.
5. Do not state that the conversation is synthetic.
6. Return only the requested JSON object.
""".strip()
    return system_prompt, user_prompt


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: ignoring malformed output line {line_number}.",
                    file=sys.stderr,
                )
                continue
            if item.get("dialogue_id"):
                completed.add(str(item["dialogue_id"]))
    return completed


def _metadata(response: LLMResponse) -> dict[str, Any]:
    return {
        "backend": response.backend,
        "model": response.model,
        "cached": response.cached,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }


def _write_jsonl(handle: Any, item: dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate profile-conditioned synthetic aphasia dialogues."
    )
    parser.add_argument("--personas", required=True)
    parser.add_argument("--output", default="outputs/dialogues.jsonl")
    parser.add_argument("--samples-per-persona", type=int, default=1)
    parser.add_argument(
        "--profile-counts",
        help="Exact quotas, e.g. anomic=100,broca=100,wernicke=50,global=25",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id-prefix", default="")
    parser.add_argument("--backend", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--cache", default="outputs/llm_cache.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=3.0)
    parser.add_argument("--request-pause", type=float, default=0.15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        quotas = parse_profile_counts(args.profile_counts)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    if not args.model and not args.dry_run:
        parser.error("--model is required unless --dry-run is used.")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1.")

    personas = load_personas(args.personas)
    jobs = plan_jobs(
        personas,
        args.samples_per_persona,
        quotas,
        args.seed,
        args.id_prefix,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.dry_run:
        print(json.dumps({
            "planned": len(jobs),
            "profile_counts": dict(
                sorted(
                    (profile, sum(job["profile"] == profile for job in jobs))
                    for profile in PROFILES
                )
            ),
            "first_jobs": [
                {
                    "dialogue_id": job["dialogue_id"],
                    "profile": job["profile"],
                    "session_topic": job["session_topic"],
                    "therapist_strategy": job["therapist_strategy"],
                }
                for job in jobs[:5]
            ],
        }, ensure_ascii=False, indent=2))
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    planned_ids = {str(job["dialogue_id"]) for job in jobs}
    completed = (
        load_completed_ids(output_path) & planned_ids if args.resume else set()
    )
    pending = [job for job in jobs if job["dialogue_id"] not in completed]
    print(
        f"Planned={len(jobs)}, completed={len(completed)}, pending={len(pending)}"
    )

    client = LLMClient(
        backend=args.backend,
        model=args.model,
        cache_path=args.cache,
    )
    output_mode = "a" if args.resume and output_path.exists() else "w"
    error_mode = "a" if args.resume and error_path.exists() else "w"
    successes = len(completed)
    failures = 0

    with output_path.open(output_mode, encoding="utf-8") as output_handle, \
            error_path.open(error_mode, encoding="utf-8") as error_handle:
        for pending_index, job in enumerate(pending, start=1):
            system_prompt, user_prompt = build_prompt(job)
            last_error: Exception | None = None

            for attempt in range(1, args.max_attempts + 1):
                started = time.time()
                try:
                    response = client.generate(
                        system_prompt,
                        user_prompt,
                        max_tokens=1200,
                        temperature=0.55,
                        bypass_cache=attempt > 1,
                    )
                    parsed = validate_dialogue_text(response.text)
                    record = {
                        "persona_id": job["persona_id"],
                        "dialogue_id": job["dialogue_id"],
                        "profile": job["profile"],
                        "session_index": job["session_index"],
                        "session_topic": job["session_topic"],
                        "therapist_strategy": job["therapist_strategy"],
                        **_metadata(response),
                        "attempt": attempt,
                        "latency_local": time.time() - started,
                        "text": json.dumps(parsed, ensure_ascii=False),
                    }
                    _write_jsonl(output_handle, record)
                    successes += 1
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[{job['dialogue_id']}] attempt {attempt}/"
                        f"{args.max_attempts} failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if attempt < args.max_attempts:
                        time.sleep(args.retry_wait * (2 ** (attempt - 1)))

            if last_error is not None:
                failures += 1
                _write_jsonl(error_handle, {
                    "persona_id": job["persona_id"],
                    "dialogue_id": job["dialogue_id"],
                    "profile": job["profile"],
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                    "time": time.time(),
                })

            processed = successes + failures
            if processed % 50 == 0 or pending_index == len(pending):
                print(
                    f"Processed {processed}/{len(jobs)}; "
                    f"success={successes}; failed={failures}"
                )
            if args.request_pause > 0:
                time.sleep(args.request_pause)

    print(f"Finished. Successful={successes}; failed={failures}.")
    print(f"Output: {output_path}")
    print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()

