from copy import deepcopy
from typing import Self, Optional

from action import Action
from game_state import GameState


class TicTacToeAction(Action):
    def __init__(self, x: int, y: int, player: int):
        self.x = x
        self.y = y
        self._player = player

    @property
    def player(self):
        return self._player

    def __repr__(self):
        return f"Action({self.x}, {self.y}, {self.player})"


class TicTacToeGameState(GameState):
    def __init__(self: Self, initial_state: Optional[list[list[Optional[int]]]] = None) -> None:
        if initial_state is None:
            self.board: list[list[Optional[bool]]] = [
                [None, None, None],
                [None, None, None],
                [None, None, None]
            ]
        else:
            self.board = initial_state

    def get_winner(self: Self) -> Optional[float]:
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] is not None:
                return float(row[0])

        for column_index in range(3):
            if self.board[0][column_index] == self.board[1][column_index] == self.board[2][column_index] and self.board[0][column_index] is not None:
                return float(self.board[0][column_index])

        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] is not None:
            return float(self.board[0][0])

        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] is not None:
            return float(self.board[0][2])

        if None not in self.board[0] and None not in self.board[1] and None not in self.board[2]:
            return 0.5

    def get_possible_actions(self: Self) -> list[Action]:
        possible_moves = []
        player = 0

        for index_row, row in enumerate(self.board):
            for index_col, col in enumerate(row):
                if col is None:
                    possible_moves.append((index_col, index_row))
                else:
                    player = 1 - player

        return [TicTacToeAction(x, y, player) for x, y in possible_moves]

    def transition(self: Self, action: TicTacToeAction) -> Self:
        new_state = TicTacToeGameState(deepcopy(self.board))
        new_state.board[action.y][action.x] = action.player

        return new_state

    def __repr__(self: Self) -> str:
        res = ""
        for row in self.board:
            res += "|".join(str(x) if x is not None else " " for x in row) + "\n"

        # Remove last \n
        return res[:-1]
