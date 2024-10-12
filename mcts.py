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

    def __init__(self: Self, parent: Optional[Self]) -> None:
        """Initialize the node.

        :param parent: Parent node in the MCTS tree.
        """
        self.parent: Optional[Self] = parent

        self.children: list[_Node] = []
        self.actions: list[Action] = []
        self.visits: int = 0
        self.n_wins: int = 0
        self.n_defeats: int = 0

    @property
    def value(self: Self) -> float:
        """Return the value of the node."""
        if self.visits == 0:
            return 0

        return (self.n_wins - self.n_defeats) / self.visits

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

    def select_child(self: Self, state: GameState) -> tuple[Self, GameState]:
        """Recursively select a leaf using the exploration/exploitation trade-off formula."""
        if self.is_leaf:
            return self, state

        best_child, best_action = max(
            zip(self.children, self.actions),
            key=lambda child_action: child_action[0].ucb,
        )

        return best_child.select_child(state.transition(best_action))

    def expand(self: Self, state: GameState) -> tuple[Self, GameState]:
        """Expand this Node by listing all the possible actions and randomly choose a child."""
        # If the state is terminal, we don't expand the associated node
        if state.get_winner() is not None:
            return self, state

        self.actions = state.get_possible_actions()
        self.children = [_Node(self) for _ in self.actions]
        chosen_child, chosen_action = choice(list(zip(self.children, self.actions)))

        return chosen_child, state.transition(chosen_action)

    @staticmethod
    def rollout(state: GameState) -> int:
        """Perform the rollout phase of the MCTS.

        This function randomly selects moves until a terminal game state is reached from this state.
        Once such a state has been reached, this function will return:
        - 1 if it resulted in a win for the player playing this position;
        - -1 if it resulted in a loss for the player playing this position;
        - 0 if it resulted in a draw.
        """
        new_state = state
        player = 1

        while (winner := new_state.get_winner()) is None:
            random_action = choice(new_state.get_possible_actions())
            new_state = new_state.transition(random_action)
            player *= -1

        return winner * player

    def backpropagate(self: Self, rollout_value: int) -> None:
        """Backpropagate the rollout result through the tree.

        This function performs the backpropagation phase of the MCTS algorithm. It updates the value
        of the nodes according to the player playing in their position and the rollout result.

        :param rollout_value: The value of the rollout result. It is equal to 1 if the current
          player won, -1 if the other player won, and 0 in case of a draw.
        """
        self.visits += 1

        # If the current player wins the rollout, then this node's value must decrease
        # This is because intuitively, the current player is the adversary of the player that will
        #   look at this node
        if rollout_value == -1:
            self.n_wins += 1
        elif rollout_value:
            self.n_defeats += 1

        # Unless we're at the root of the tree
        if self.parent is not None:
            self.parent.backpropagate(-rollout_value)

    def __repr__(self: Self) -> str:
        value = self.value
        visits = self.visits

        return f"_Node({value=}, {visits=})"


class MCTS:
    """A class used to perform an MCTS algorithm."""

    def __init__(
        self: Self,
        state: GameState,
        trade_off_constant: float,
        n_simulations: int,
    ) -> None:
        """Initialize the MCTS algorithm.

        This function sets the global parameters used by the MCTS algorithm and initialize it.

        :param state: The state the game starts in.
        :param trade_off_constant: The trade-off constant as used in the UCB formula.
        :param n_simulations: The number of simulations that are performed from the root.
        """
        _Node.set_trade_off_constant(trade_off_constant)
        self.root = _Node(None)
        self.root_state = state
        self.n_simulations = n_simulations

    def transition(self: Self, action: Action) -> None:
        """Advance the game with a given action.

        This function allows to progress in the game tree without any computation. This may for
        example prove useful if the adversary is exterior to this class.
        """
        index = self.root.actions.index(action)

        self.root = self.root.children[index]
        self.root_state = self.root_state.transition(action)

    def decide(self: Self, advance: bool = True) -> Optional[Action]:
        """Decide the next move to be played.

        This function performs a number of simulation as specified in the algorithm initialization,
        updating the nodes' scores along the way. Once these simulations have been performed, it
        selects the best child to go with according to its visit counts.

        :param advance: If set to True, in addition to returning the best action, the root will be
          set to the child corresponding to this action.
        """
        node = self.root

        for _ in trange(self.n_simulations):
            node, state = node.select_child(self.root_state)
            node, state = node.expand(state)
            rollout_value = node.rollout(state)
            node.backpropagate(rollout_value)
            node = self.root

        chosen_action = self.root.actions[
            max(enumerate(self.root.children), key=lambda x: x[1].visits)[0]
        ]

        if advance:
            self.transition(chosen_action)

        return chosen_action
