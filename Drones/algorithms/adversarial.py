from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """

        def minimax(state: GameState, agent_index: int, depth: int) -> float:
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            legal_actions = state.get_legal_actions(agent_index)

            if not legal_actions:
                return self.evaluation_function(state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            if agent_index == 0:
                best = float("-inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = minimax(successor, next_agent, next_depth)
                    best = max(best, val)
                return best
            else:
                best = float("inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = minimax(successor, next_agent, next_depth)
                    best = min(best, val)
                return best

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        num_agents = state.get_num_agents()
        best_action = None
        best_val = float("-inf")

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = minimax(successor, next_agent, next_depth)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """

        def alphabeta(
            state: GameState,
            agent_index: int,
            depth: int,
            alpha: float,
            beta: float,
        ) -> float:
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            legal_actions = state.get_legal_actions(agent_index)

            if not legal_actions:
                return self.evaluation_function(state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            if agent_index == 0:
                val = float("-inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = max(val, alphabeta(successor, next_agent, next_depth, alpha, beta))
                    if val > beta:  
                        return val
                    alpha = max(alpha, val)
                return val
            else:
                val = float("inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = min(val, alphabeta(successor, next_agent, next_depth, alpha, beta))
                    if val < alpha: 
                        return val
                    beta = min(beta, val)
                return val

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        num_agents = state.get_num_agents()
        best_action = None
        best_val = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = alphabeta(successor, next_agent, next_depth, alpha, beta)
            if val > best_val:
                best_val = val
                best_action = action
            alpha = max(alpha, best_val)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """

        def expectimax(state: GameState, agent_index: int, depth: int) -> float:
            if state.is_win() or state.is_lose() or depth == 0:
                return self.evaluation_function(state)

            num_agents = state.get_num_agents()
            legal_actions = state.get_legal_actions(agent_index)

            if not legal_actions:
                return self.evaluation_function(state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth - 1 if next_agent == 0 else depth

            if agent_index == 0:
                best = float("-inf")
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    val = expectimax(successor, next_agent, next_depth)
                    best = max(best, val)
                return best
            else:
                child_values = []
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    child_values.append(expectimax(successor, next_agent, next_depth))

                min_val = min(child_values)
                avg_val = sum(child_values) / len(child_values)
                p = self.prob
                return (1 - p) * min_val + p * avg_val

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        num_agents = state.get_num_agents()
        best_action = None
        best_val = float("-inf")

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            next_agent = 1 % num_agents
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = expectimax(successor, next_agent, next_depth)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action
