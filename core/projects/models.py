"""Project data model and state machine. See DESIGN.md S6-S7."""

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ProjectState(Enum):
    CREATED = "created"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    TRAINING = "training"
    TRAINED = "trained"
    ERROR = "error"


VALID_TRANSITIONS: dict[ProjectState, frozenset] = {
    ProjectState.CREATED: frozenset({ProjectState.PREPROCESSING}),
    ProjectState.PREPROCESSING: frozenset({ProjectState.PREPROCESSED, ProjectState.ERROR}),
    ProjectState.PREPROCESSED: frozenset({ProjectState.PREPROCESSING, ProjectState.TRAINING}),
    ProjectState.TRAINING: frozenset({ProjectState.TRAINED, ProjectState.ERROR}),
    ProjectState.TRAINED: frozenset({ProjectState.TRAINING}),
    ProjectState.ERROR: frozenset({ProjectState.PREPROCESSING, ProjectState.TRAINING}),
}


class InvalidTransitionError(ValueError):
    def __init__(self, from_state: ProjectState, to_state: ProjectState):
        super().__init__(f"Cannot transition from {from_state.value!r} to {to_state.value!r}")
        self.from_state = from_state
        self.to_state = to_state


def now_iso(now=None) -> str:
    moment = now() if now is not None else datetime.now(timezone.utc)
    return moment.isoformat()


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    state: ProjectState
    created_at: str
    updated_at: str
    language: str = "en"
    config: dict = field(default_factory=dict)
    error_message: str | None = None
    error_from_state: ProjectState | None = None

    @classmethod
    def new(
        cls, *, id: str, name: str, language: str = "en", config: dict | None = None, now=None
    ) -> "Project":
        timestamp = now_iso(now)
        return cls(
            id=id,
            name=name,
            state=ProjectState.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
            language=language,
            config=dict(config) if config else {},
        )

    def transition_to(
        self, new_state: ProjectState, *, error_message: str | None = None, now=None
    ) -> "Project":
        allowed = VALID_TRANSITIONS.get(self.state, frozenset())
        if new_state not in allowed:
            raise InvalidTransitionError(self.state, new_state)

        if new_state is ProjectState.ERROR:
            return dataclasses.replace(
                self,
                state=new_state,
                updated_at=now_iso(now),
                error_message=error_message,
                error_from_state=self.state,
            )

        return dataclasses.replace(
            self,
            state=new_state,
            updated_at=now_iso(now),
            error_message=None,
            error_from_state=None,
        )
