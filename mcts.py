from math import sqrt, log
from random import choice
from typing import Self, Optional

from tqdm import trange

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

    def select_child(self: Self, player: int) -> tuple[Self, int]:
        """Recursively select a leaf using the exploration/exploitation trade-off formula."""
        if self.is_leaf:
            return self, player

        scores = [(index, child.ucb) for index, child in enumerate(self.children)]
        scores.sort(key=lambda x: x[1], reverse=True)

        return self.children[choice([index for index, ucb in scores if ucb == scores[0][1]])].select_child(1 - player)

    def expand(self: Self, player: int) -> tuple[Self, int]:
        """Expand this Node by listing all the possible actions and choose a child."""
        # If the state is terminal, we don't expand the associated node
        if self.state.get_winner() is not None:
            return self, player

        assert self.is_leaf, "Patate"

        actions = self.state.get_possible_actions()
        self.children = [
            _Node(self, action, self.state.transition(action)) for action in actions
        ]

        return choice(self.children), 1 - player

    def rollout(self: Self) -> float:
        """Perform the rollout phase of the MCTS.

        This function randomly selects moves until a terminal game state is reached from this state.
        Once such a state has been reached, this function will return:
        - 0 if it resulted in a win for the first player;
        - 1 if it resulted in a loss for the first player;
        - 0.5 if it resulted in a draw.
        """
        winner = self.state.get_winner()
        new_state = self.state

        while winner is None:
            random_action = choice(new_state.get_possible_actions())
            new_state = new_state.transition(random_action)
            winner = new_state.get_winner()

        return winner

    def backpropagate(self: Self, rollout_value: float, player_to_play: int) -> None:
        """Backpropagate the rollout result through the tree."""
        self.visits += 1

        if rollout_value == 0.5:
            self.n_draws += 1
        # If the first player win the rollout, while the second player has to play this position,
        # then it means that the node's value has to increase
        elif rollout_value != player_to_play:
            self.n_wins += 1

        # Unless we're at the root of the tree
        if self.parent is not None:
            self.parent.backpropagate(rollout_value, 1 - player_to_play)

    def __repr__(self: Self) -> str:
        return f"_Node({self.value=}, {self.state.__repr__()})"


class MCTS:
    def __init__(
        self: Self,
        state: GameState,
        trade_off_constant: float,
        n_simulations: int,
        player: Optional[int] = None
    ) -> None:
        _Node.set_trade_off_constant(trade_off_constant)
        self.root = _Node(None, None, state)
        self.n_simulations = n_simulations

        if player is None:
            player = 0

        if player not in [0, 1]:
            raise ValueError("Player must be 0 or 1.")

        self.player = player

    def get_decision(self: Self) -> Action:
        node = self.root

        for i in trange(self.n_simulations):
            node, player = node.select_child(self.player)
            node, player = node.expand(player)
            rollout_value = node.rollout()
            node.backpropagate(rollout_value, player)
            node = self.root

        values = [(index, child.value) for index, child in enumerate(self.root.children)]
        values.sort(key=lambda x: x[1], reverse=True)

        # We pick the first one instead of picking randomly in case of an equality
        return self.root.children[values[0][0]].action

    def set_new_root(self: Self, new_root: _Node) -> None:
        self.root = new_root
