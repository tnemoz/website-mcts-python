from abc import ABC, abstractmethod
from typing import Optional, Self

from action import Action


class GameState(ABC):
    """Abstract class to represent a game state."""

    @abstractmethod
    def is_terminal(self: Self) -> bool:
        """Return True if the current game state is a terminal one."""
        pass

    @abstractmethod
    def get_possible_actions(self: Self) -> list[Action]:
        """Return the list of possible actions when playing this game state."""
        pass

    @abstractmethod
    def transition(self: Self, action: Action) -> tuple[Self, Optional[float]]:
        """Return the new game state along with the potential winner of the game from an action.

        Return the game state resulting from a given action on the current game state along with the
        potential winner. That is, if the resulting game state isn't terminal, then None is returned
        to represent the winner. Otherwise, it returns:
        - 1 if the player who just played won;
        - 0 if the player who just played lost;
        - 0.5 otherwise.
        """
        pass
