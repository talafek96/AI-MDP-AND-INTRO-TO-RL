from copy import deepcopy
import random
from typing import List, Set, Tuple
import numpy as np
from mdp import MDP


def get_states(mdp: MDP) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(len(mdp.board)) for j in range(len(mdp.board[0])) if mdp.board[i][j] != 'WALL']


def get_reward(mdp: MDP, state: Tuple[int, int]) -> float:
    return float(mdp.board[state[0]][state[1]]) if mdp.board[state[0]][state[1]] != 'WALL' else -np.inf


def get_actions(mdp: MDP):
    return list(mdp.transition_function.keys())


def get_neighbours(mdp: MDP, state: Tuple[int, int]) -> List[Tuple[int, int]]:
    return list({mdp.step(state, action) for action in get_actions(mdp)})


def get_probability(mdp: MDP, state: Tuple[int, int], action: str, dest_state: Tuple[int, int]):
    '''
    Implements P(dest_state | action, state)
    '''
    actions = get_actions(mdp)
    assert action in actions

    if state in mdp.terminal_states or dest_state not in get_neighbours(mdp, state):
        return 0.

    action_to_index = dict(zip(actions, range(len(actions))))
    deter_actions_to_dest_state = [action for action in actions if mdp.step(state, action) == dest_state]

    return sum(float(mdp.transition_function[action][action_to_index[orientation_action]]) \
                for orientation_action in deter_actions_to_dest_state)


def value_iteration(mdp: MDP, U_init: List[List[float]], epsilon: float=1e-3):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    states = get_states(mdp)
    actions = get_actions(mdp)
    delta = np.inf
    U_next = deepcopy(U_init)
    U = deepcopy(U_init)

    while delta >= epsilon * (1 - mdp.gamma) / mdp.gamma:
        U = deepcopy(U_next)
        delta = 0

        for state in states:
            U_next[state[0]][state[1]] = \
                get_reward(mdp, state) + \
                    mdp.gamma * max([
                            sum(get_probability(mdp, state, action, neighbour) * U[neighbour[0]][neighbour[1]] \
                                for neighbour in get_neighbours(mdp, state)) \
                                    for action in actions
                        ])
            
            delta = max(delta, np.abs(U_next[state[0]][state[1]] - U[state[0]][state[1]]))
    
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
