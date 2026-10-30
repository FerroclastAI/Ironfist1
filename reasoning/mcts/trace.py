
"""Structured reasoning traces for IronFist1's MCTS."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCTSTraceStep:
    """One recorded event during an MCTS search."""

    state: str
    depth: int
    event: str
    reward: float | None = None
    prior: float | None = None
    selected: bool = False


@dataclass
class MCTSTrace:
    """Complete trace of an MCTS reasoning run."""

    steps: list[MCTSTraceStep] = field(default_factory=list)

    def record(
        self,
        state: str,
        depth: int,
        event: str,
        reward: float | None = None,
        prior: float | None = None,
        selected: bool = False,
    ) -> None:
        if depth < 0:
            raise ValueError("depth cannot be negative")

        self.steps.append(
            MCTSTraceStep(
                state=state,
                depth=depth,
                event=event,
                reward=reward,
                prior=prior,
                selected=selected,
            )
        )

    def mark_selected(self, state: str) -> None:
        """Mark the final selected state."""
        for step in self.steps:
            if step.state == state:
                step.selected = True

    @property
    def selected_steps(self) -> list[MCTSTraceStep]:
        return [step for step in self.steps if step.selected]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "steps": [
                {
                    "state": step.state,
                    "depth": step.depth,
                    "event": step.event,
                    "reward": step.reward,
                    "prior": step.prior,
                    "selected": step.selected,
                }
                for step in self.steps
            ]
        }
