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

    def __init__(self: Self, parent: Optional[Self], state: GameState) -> None:
        """Initialize the node.

        :param parent: Parent node in the MCTS tree.
        :param state: The game state this node represents.
        """
        self.parent: Optional[Self] = parent
        self.state: GameState = state

        self.children: list[_Node] = []
        self.actions: list[Action] = []
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
        return self.value + self.C * sqrt(log(self.parent.visits) / self.visits)

    @property
    def is_leaf(self: Self) -> bool:
        """Return True if this Node is a leaf node."""
        return len(self.children) == 0

    def select_child(self: Self, player: int) -> tuple[Self, int]:
        """Recursively select a leaf using the exploration/exploitation trade-off formula."""
        if self.is_leaf:
            return self, player

        return max(self.children, key=lambda child: child.ucb).select_child(1 - player)

    def expand(self: Self, player: int) -> tuple[Self, int]:
        """Expand this Node by listing all the possible actions and randomly choose a child."""
        # If the state is terminal, we don't expand the associated node
        if self.state.get_winner() is not None:
            return self, player

        self.actions = self.state.get_possible_actions()
        self.children = [
            _Node(self, self.state.transition(action)) for action in self.actions
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
        new_state = self.state

        while (winner := new_state.get_winner()) is None:
            random_action = choice(new_state.get_possible_actions())
            new_state = new_state.transition(random_action)

        return winner

    def backpropagate(self: Self, rollout_value: float, player_to_play: int) -> None:
        """Backpropagate the rollout result through the tree.

        This function performs the backpropagation phase of the MCTS algorithm. It updates the value
        of the nodes according to the player playing in their position and the rollout result.

        :param rollout_value: The value of the rollout result. It is equal to 0 if the first player
          won, 1 if the second player won, and 0.5 in case of a draw.
        :param player_to_play: The player having to play the position represented by this node.
        """
        self.visits += 1

        if rollout_value == 0.5:
            self.n_draws += 1
        # If the first player wins the rollout, while the second player has to play this position,
        # then it means that the node's value has to increase
        elif rollout_value != player_to_play:
            self.n_wins += 1

        # Unless we're at the root of the tree
        if self.parent is not None:
            self.parent.backpropagate(rollout_value, 1 - player_to_play)

    def __repr__(self: Self) -> str:
        value = self.value
        visits = self.visits

        return f"_Node({value=}, {visits=}, {self.state.__repr__()})"


class MCTS:
    """A class used to perform an MCTS algorithm."""
    def __init__(
        self: Self,
        state: GameState,
        trade_off_constant: float,
        n_simulations: int,
        player: Optional[int] = None
    ) -> None:
        """Initialize the MCTS algorithm.

        This function sets the global parameters used by the MCTS algorithm and initialize it.

        :param state: The state the game starts in.
        :param trade_off_constant: Teh trade-off constant as used in the UCB formula.
        :param n_simulations: The number of simulations that are performed from the root.
        :param player: The player that plays the position represented by the root.
        """
        _Node.set_trade_off_constant(trade_off_constant)
        self.root = _Node(None, state)
        self.n_simulations = n_simulations

        if player is None:
            player = 0

        if player not in [0, 1]:
            raise ValueError("Player must be 0 or 1.")

        self.player = player

    def decide(self: Self, advance: bool = True) -> Optional[Action]:
        """Decide the next move to be played.

        This function performs a number of simulation as specified in the algorithm initialization,
        updating the nodes' scores along the way. Once these simulations have been performed, it
        selects the best child to go with according to its visit counts.

        :param advance: If set to True, nothing will be returned and the new root will be set to the
          aforementioned child. Otherwise, it returns the action leading to the best child.
        """
        node = self.root

        for _ in trange(self.n_simulations):
            node, player = node.select_child(self.player)
            node, player = node.expand(player)
            rollout_value = node.rollout()
            node.backpropagate(rollout_value, player)
            node = self.root

        chosen_index = max(enumerate(self.root.children), key=lambda x: x[1].visits)[0]

        if advance:
            self.root = self.root.children[chosen_index]
            self.player = 1 - self.player

        return self.root.actions[chosen_index]
