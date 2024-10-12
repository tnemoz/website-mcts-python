from typing import Self, Optional

import numpy as np
import numpy.typing as npt

from action import Action
from game_state import GameState


class TicTacToeAction(Action):
    """A Tic-Tac-Toe action."""

    def __init__(self: Self, x: int, y: int) -> None:
        """Initialize the action."""
        self.x = x
        self.y = y

    def __eq__(self: Self, other: Self):
        return self.x == other.x and self.y == other.y

    def __hash__(self: Self):
        return 3 * self.x + self.y

    def __repr__(self: Self) -> str:
        """Return a string representation of the action."""
        return f"Action({self.x}, {self.y})"


class TicTacToeGameState(GameState):
    """A Tic-Tac-Toe game state."""

    def __init__(
        self: Self, initial_state: Optional[npt.NDArray[np.int8]] = None
    ) -> None:
        """Initialize the game state."""
        if initial_state is None:
            self.board: npt.NDArray[np.int8] = np.zeros((3, 3), dtype=np.int8)
        else:
            if initial_state.shape != (3, 3):
                raise ValueError(
                    "Wrong shape provided for initial state. "
                    f"Expected (3,3) and got {initial_state.shape}"
                )

            self.board = initial_state

    def get_winner(self: Self) -> Optional[int]:
        check_col = self.board.sum(axis=0)
        check_row = self.board.sum(axis=1)
        check_diag = np.trace(self.board)
        check_anti_diag = np.trace(np.fliplr(self.board))

        if -3 in check_col or -3 in check_row or check_diag == -3 or check_anti_diag == -3:
            return -1

        if 0 not in self.board:
            return 0

    def get_possible_actions(self: Self) -> list[Action]:
        return [
            TicTacToeAction(int(y), int(x)) for x, y in np.argwhere(self.board == 0)
        ]

    def transition(self: Self, action: TicTacToeAction) -> Self:
        new_state = TicTacToeGameState(self.board.copy())
        new_state.board[action.y, action.x] = 1
        new_state.board *= -1

        return new_state

    def __repr__(self: Self) -> str:
        """Return a string representation of the game state."""
        res = ""
        for row in self.board:
            res += "|".join(("X" if x == 1 else "O") if x else " " for x in row) + "\n"

        # Remove last \n
        return res[:-1]
