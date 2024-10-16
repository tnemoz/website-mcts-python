import numpy as np

from mcts import MCTS
from tictactoe import TicTacToeGameState

initial_state = np.array(
    [
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
)

mcts_player = MCTS(TicTacToeGameState(initial_state), np.sqrt(2), 800)
action = mcts_player.decide(advance=False)

print(mcts_player.root_state.transition(action))
