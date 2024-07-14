from abc import ABC, abstractmethod
from typing import Optional, Self

from action import Action


class GameState(ABC):
    """Abstract class to represent a game state."""

    @abstractmethod
    def get_winner(self: Self) -> Optional[float]:
        """Return the winner of the game, if it exists.

        Return the game state resulting from a given action on the current game state along with the
        potential winner. That is, if the resulting game state isn't terminal, then None is returned
        to represent the winner. Otherwise, it returns:
        - 0 if the first player won;
        - 1 if the second player won;
        - 0.5 in case of a draw.
        """
        pass

    @abstractmethod
    def get_possible_actions(self: Self) -> list[Action]:
        """Return the list of possible actions when playing this game state."""
        pass

    @abstractmethod
    def transition(self: Self, action: Action) -> Self:
        """Return the new game state resulting from a given action applied on the current state."""
        pass
