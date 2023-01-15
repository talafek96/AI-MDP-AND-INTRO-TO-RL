from copy import deepcopy
import random
from typing import List, Set, Tuple
import numpy as np
from mdp import MDP


def get_states(mdp: MDP) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(len(mdp.board)) for j in range(len(mdp.board[0])) if mdp.board[i][j] != 'WALL']


def is_state(mdp: MDP, state: Tuple[int, int]) -> bool:
    return state in get_states(mdp)


def get_reward(mdp: MDP, state: Tuple[int, int]) -> float:
    return float(mdp.board[state[0]][state[1]]) if is_state(mdp, state) else -np.inf


def get_actions(mdp: MDP):
    return list(mdp.transition_function.keys())


def get_neighbours(mdp: MDP, state: Tuple[int, int]) -> List[Tuple[int, int]]:
    return list({mdp.step(state, action) for action in get_actions(mdp)})


def simulate_step(mdp: MDP, state: Tuple[int, int], action: str) -> Tuple[int, int]:
    '''
    Simulates a step in the world from `state` with the `action` given,
    and returns the random result using the transition function of the MDP.
    '''
    actions = get_actions(mdp)
    action_to_index = dict(zip(actions, range(len(actions))))

    next_state_probabilities = {next_state: probability 
                                    for next_state, probability in \
                                        [(mdp.step(state, gt_action), mdp.transition_function[action][action_to_index[gt_action]]) \
                                            for gt_action in actions]}
    next_state = random.choices(list(next_state_probabilities.keys()), weights=list(next_state_probabilities.values()), k=1)[0]
    return next_state


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


def get_policy(mdp: MDP, U: List[List[float]]):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    actions = get_actions(mdp)
    index_to_action = dict(enumerate(actions))
    def get_best_action(state: Tuple[int, int]) -> str:
        nonlocal mdp, U, index_to_action, actions
        assert state in get_states(mdp)

        best_action = index_to_action[np.argmax([sum(get_probability(mdp, state, action, neighbour) * U[neighbour[0]][neighbour[1]]
                                        for neighbour in get_neighbours(mdp, state))
                                            for action in actions])]
        return best_action

    return [[get_best_action((i, j)) if is_state(mdp, (i, j)) else None for j in range(mdp.num_col)] for i in range(mdp.num_row)]
    # ========================


def q_learning(mdp: MDP, init_state: Tuple[int, int], total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8, show_progress: bool=False):
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
    states = get_states(mdp)
    actions = get_actions(mdp)
    action_to_index = dict(zip(actions, range(len(actions))))
    index_to_action = dict(enumerate(actions))
    state_to_index = dict(zip(states, range(len(states))))

    qtable = np.zeros((len(states), len(actions)))

    for episode in range(total_episodes):
        # Reset the environment
        state = deepcopy(init_state)
        done = False

        for _ in range(max_steps):
            # Choose an action in the current state.
            ## First, sample a random number to determine whether to explore or exploit
            explore_exploit_tradeoff = random.uniform(0, 1)

            ## If it is greater than epsilon --> exploit (take the best/biggest Q value for this state)
            if explore_exploit_tradeoff > epsilon:
                chosen_action = index_to_action[np.argmax(qtable[state_to_index[state], :])]

            ## Else --> explore (do a random action)
            else:
                chosen_action = random.sample(actions, 1)[0]
            
            # Take the chosen action and observe the outcome state and the corresponding reward
            new_state = simulate_step(mdp, state, chosen_action)
            reward = get_reward(mdp, new_state)
            done = True if new_state in mdp.terminal_states else False

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max (Q(s',a')) - Q(s,a)]
            qtable[state_to_index[state], action_to_index[chosen_action]] = \
                qtable[state_to_index[state], action_to_index[chosen_action]] + learning_rate * (reward + mdp.gamma *
                    np.max(qtable[state_to_index[new_state], :]) - qtable[state_to_index[state], action_to_index[chosen_action]])
            
            # Update the current state
            state = deepcopy(new_state)

            # Finish the episode if reached a terminal state
            if done:
                break
        
        # Reduce epsilon in order to get less and less exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    
    return qtable
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    states = get_states(mdp)
    actions = get_actions(mdp)
    state_to_index = dict(zip(states, range(len(states))))
    index_to_action = dict(enumerate(actions))

    def get_best_action(state: Tuple[int, int]) -> str:
        nonlocal qtable, state_to_index, index_to_action
        return index_to_action[np.argmax(qtable[state_to_index[state], :])]
    
    return [[get_best_action((i, j)) if is_state(mdp, (i, j)) else None for j in range(mdp.num_col)] for i in range(mdp.num_row)]
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
