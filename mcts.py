from math import sqrt, log
from random import choice
from typing import Self, Optional

from action import Action
from game_state import GameState


class _Node:
    """Represent a node in the MCTS tree."""

    # Constant that represents the exploration/exploitation trade-off
    C: float = sqrt(2)

    @classmethod
    def set_trade_off_constant(cls: type(Self), value: float) -> None:
        """Update the value of the exploration/exploitation trade-off constant."""
        cls.C = value

    def __init__(
        self: Self,
        parent: Optional[Self],
        action: Optional[Action],
        state: GameState
    ) -> None:
        """Initialize the node."""
        self.parent: Optional[Self] = parent
        self.action: Optional[Action] = action
        self.state: GameState = state
        self.children: list[_Node] = []
        self.visits: int = 0
        self.n_wins: int = 0
        self.n_draws: int = 0

    @property
    def value(self: Self) -> float:
        """Return the value of the node."""
        if self.visits == 0:
            return 0

        return (2 * self.n_wins + self.n_draws) / (2 * self.visits)

    @property
    def ucb(self: Self) -> float:
        """Return the USB score of this Node."""
        if self.visits == 0:
            return float("inf")

        # No need to check for the parent's visit count to be positive, since it is necessarily
        # larger than that of its child
        return self.value + self.C * sqrt(log(self.parent.visits + 1) / self.visits)

    @property
    def is_leaf(self: Self) -> bool:
        """Return True if this Node is a leaf node."""
        return len(self.children) == 0

    def select_child(self: Self) -> Self:
        """Recursively select a leaf using the exploration/exploitation trade-off formula."""
        if self.is_leaf:
            return self

        scores = [(index, child.ucb) for index, child in enumerate(self.children)]
        scores.sort(key=lambda x: x[1], reverse=True)

        # We pick the first one instead of picking randomly in case of an equality
        return self.children[scores[0][0]]

    def expand(self: Self) -> None:
        """Expand this Node by listing all the possible actions."""
        # If the state is terminal, we don't expand the associated node
        if self.state.is_terminal():
            return

        actions = self.state.get_possible_actions()
        self.children = [
            _Node(self, action, self.state.transition(action)[0]) for action in actions
        ]

    def rollout(self: Self) -> float:
        """Perform the rollout phase of the MCTS.

        This function randomly selects moves until a terminal game state is reached from this state.
        Once such a state has been reached, this function will return:
        - 0 if it resulted in a win for the player to play starting from that position;
        - 1 if it resulted in a loss for the player to play starting from that position;
        - 0.5 if it resulted in a draw.

        That is, this function will return the value that must be added to the node's value
        representing this state. So, since the adversary will play that position, we want to return
        0 if they win, and 1 if we do.
        """
        # If the state is terminal, its outcome will be the same as during its first iteration
        if self.state.is_terminal():
            if self.n_wins > 0:
                return 1

            if self.n_draws > 0:
                return 0.5

            return 0

        # We have to keep track of whose player turn it is. Indeed, it will determine whether we
        # add or subtract the result from the Node's value. Since our goal is to evaluate whether
        # this node is an acceptable state for the player that has to make a decision, it means that
        # starting from this state, the first player to move is the adversary.
        player = 1.
        random_action = choice(self.state.get_possible_actions())
        new_state, winner = self.state.transition(random_action)

        while winner is None:
            player = 1 - player
            random_action = choice(self.state.get_possible_actions())
            new_state, winner = self.state.transition(random_action)

        # The winner is the last one having played
        if winner == 1:
            # 0 if the adversary is the last one to have played, 1 otherwise
            return 1 - player

        if winner == 0:
            # 0 if the adversary is the last one to have played, 1 otherwise
            return player

        return winner

    def backpropagate(self: Self, rollout_value: float) -> None:
        """Backpropagate the rollout result through the tree."""
        self.visits += 1

        if rollout_value == 1:
            self.n_wins += 1
        elif rollout_value == 0.5:
            self.n_draws += 1

        # Unless we're at the root of the tree
        if self.parent is not None:
            self.parent.backpropagate(1 - rollout_value)


class MCTS:
    def __init__(
        self: Self,
        state: GameState,
        trade_off_constant: float,
        n_simulations: int
    ) -> None:
        _Node.set_trade_off_constant(trade_off_constant)
        self.root = _Node(None, None, state)
        self.n_simulations = n_simulations

    def get_decision(self: Self) -> Action:
        node = self.root

        for i in range(self.n_simulations):
            node = node.select_child()
            node.expand()
            rollout_value = node.rollout()
            node.backpropagate(rollout_value)
            node = self.root

        values = [(index, child.value) for index, child in enumerate(self.root.children)]
        values.sort(key=lambda x: x[1], reverse=True)

        # We pick the first one instead of picking randomly in case of an equality
        return self.children[values[0][0]]
