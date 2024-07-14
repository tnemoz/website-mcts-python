from abc import ABC, abstractmethod
from typing import Self


class Action(ABC):
    """Represent a generic possible action a player can do in a game."""

    @property
    @abstractmethod
    def player(self: Self) -> int:
        """Represents the player executing this action."""
        pass
