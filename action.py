from abc import ABC, abstractmethod
from typing import Self


class Action(ABC):
    """Represent a generic possible action a player can do in a game."""

    @abstractmethod
    def __eq__(self: Self, other: Self) -> bool:
        """Test equality between two Actions."""
        raise pass

    @abstractmethod
    def __hash__(self: Self) -> int:
        """Return a unique hash value for this Action."""
        raise pass
