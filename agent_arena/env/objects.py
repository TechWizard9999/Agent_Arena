from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """Simple coordinate container using (x, y) indexing."""

    x: int
    y: int

    def moved(self, dx: int, dy: int) -> "Position":
        return Position(self.x + dx, self.y + dy)

    def manhattan(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def as_tuple(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass
class ArenaLayout:
    """Holds the mutable layout state for a single episode."""

    agent: Position
    key: Position
    door: Position
    goal: Position
    structural_walls: set[Position]
    obstacles: set[Position]


class Tokens:
    """ASCII tokens used by the console renderer."""

    EMPTY = "."
    AGENT = "A"
    KEY = "K"
    CLOSED_DOOR = "D"
    OPEN_DOOR = "O"
    GOAL = "G"
    STRUCTURAL_WALL = "#"
    OBSTACLE = "X"

