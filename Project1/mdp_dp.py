### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
# Modified By Yanhua Li on 09/09/2022 for gym==0.25.2
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
# Updated for DS551/CS551 Fall 2026 with gymnasium==1.2.2
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gymnasium.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    terminate = False
    while not terminate:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            value_function[s] = sum(policy[s][a] * sum(probability * (reward + gamma * value_function[next_state] * (not terminal)) for probability, next_state, reward, terminal in P[s][a]) for a in range(nA))
            delta = max(delta, abs(v - value_function[s]))
        if delta < tol:
            terminate = True
    

    return value_function 


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution
	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################

    for s in range(nS):
        # Calculate the action values for each action in state s
        action_values = np.zeros(nA)
        for a in range(nA):
            action_values[a] = sum(probability * (reward + gamma * value_from_policy[next_state]) for probability, next_state, reward, terminal in P[s][a])
        
        # Find the best action(s) and update the policy
        best_action = np.argmax(action_values)
        new_policy[s] = np.eye(nA)[best_action]  # Set the best action to 1 and others to 0
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    
	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################
    
    while True:
        V = policy_evaluation(P, nS, nA, new_policy, gamma, tol)
        updated_policy = policy_improvement(P, nS, nA, V, gamma)
        if np.array_equal(updated_policy, new_policy):
            break
        new_policy = updated_policy

    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    V_new = V.copy()
    policy_new = np.zeros([nS, nA])
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    terminate = False
    while not terminate:
        delta = 0
        for s in range(nS):
            v = V_new[s]
            V_new[s] = max(sum(probability * (reward + gamma * V_new[next_state] * (not terminal)) for probability, next_state, reward, terminal in P[s][a]) for a in range(nA))
            delta = max(delta, abs(v - V_new[s]))
        if delta < tol:
            terminate = True

    # After convergence, derive the policy from the value function
    for s in range(nS):
        action_values = np.zeros(nA)
        for a in range(nA):
            action_values[a] = sum(probability * (reward + gamma * V_new[next_state] * (not terminal)) for probability, next_state, reward, terminal in P[s][a])
        best_action = np.argmax(action_values)
        policy_new[s] = np.eye(nA)[best_action]  # Set the best action to 1 and others to 0

    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game environment from gymnasium, play multiple episodes of the game.
    An episode is over when the returned value for "terminated" or "truncated" is True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have observation_space, action_space, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob, _ = env.reset() # initialize the episode
        done = False
        while not done: # using "not truncated" as well, when using time_limited wrapper.
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            #                          #
            ############################
            action = np.argmax(policy[ob]) # pick the action with the highest probability
            ob, reward, terminated, truncated, info = env.step(action) # take the action
            done = terminated or truncated # check if the episode is over
            total_rewards += reward # accumulate the reward
    env.close() # close the environment

            
    return total_rewards


