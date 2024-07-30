from abc import ABC, abstractmethod
from typing import Optional, Self

from action import Action


class GameState(ABC):
    """Abstract class to represent a game state.

    It is assumed that the player playing the position can be identified using some internal data.
    For instance, the player whose turn it is could have their pieces represented by positive
    numbers, while the other one would have their pieces represented by negative numbers.
    """

    @abstractmethod
    def get_winner(self: Self) -> Optional[float]:
        """Return the winner of the game, if it exists.

        If the current game state isn't terminal, then None is returned. Otherwise, it returns:
        - 1 if player playing this position won.;
        - -1 if the other player won;
        - 0 in case of a draw.
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
