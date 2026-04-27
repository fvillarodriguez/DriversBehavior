from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    uuid: str
    name: str
    host: str
    port: int
    preferred: bool = False


def choose_scheduler(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (not item.preferred, item.uuid))[0]


def should_self_promote(self_candidate: Candidate, seen: list[Candidate]) -> bool:
    winner = choose_scheduler([self_candidate, *seen])
    return winner == self_candidate

