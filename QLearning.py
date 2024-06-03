import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

# matplotlib.use('Agg')


# Read parameters from data2.txt file

def read_parameters_from_file(file_path):
    parameters = {
        'N': None,
        'M': None,
        'p1': None,
        'p2': None,
        'p3': None,
        'r': None,
        'gamma': 0.99,
        'terminals': [],
        'special_states': [],
        'forbidden_states': [],
        'start_state': [],
        'epsilon': 0.05
    }
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # Skip lines with less than 2 elements

                label = parts[0]
                if label == 'N':
                    parameters['N'] = int(parts[1])
                elif label == 'M':
                    parameters['M'] = int(parts[1])
                elif label == 'P':
                    parameters['p1'] = float(parts[1])
                    parameters['p2'] = float(parts[2])
                    parameters['p3'] = float(parts[3])
                elif label == 'R':
                    parameters['r'] = float(parts[1])
                elif label == 'G':
                    parameters['gamma'] = float(parts[1])
                elif label == 'T':
                    parameters['terminals'].append(((int(parts[1]), int(parts[2])), float(parts[3])))
                elif label == 'B':
                    parameters['special_states'].append(((int(parts[1]), int(parts[2])), float(parts[3])))
                elif label == 'F':
                    parameters['forbidden_states'].append((int(parts[1]), int(parts[2])))
                elif label == 'S':
                    parameters['start_state'] = (int(parts[1]), int(parts[2]))
                elif label == 'E':
                    parameters['epsilon'] = float(parts[1])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print("An error occurred while reading the data file:", e)
    return parameters


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Q-learning on a grid world.')
parser.add_argument('--iterations', type=int,default=10000, help='Number of iterations for the Q-learning algorithm')
parser.add_argument('--epsilon', type=float, default=0.05, help='Epsilon value for the Q-learning algorithm')
parser.add_argument('--data_file', type=str, default="4x4.txt", help='Path to the data file containing grid world parameters')
args = parser.parse_args()

# Define the grid world parameters
parameters = read_parameters_from_file(args.data_file)
N = parameters['N']
M = parameters['M']
p1 = parameters['p1']
p2 = parameters['p2']
p3 = parameters['p3']
r = parameters['r']
gamma = parameters['gamma']
epsilon = args.epsilon

# Define rewards for terminal states
rewards = np.full((N, M), r)
for terminal, reward in parameters.get('terminals', []):
    rewards[terminal] = reward

# Define rewards for special states
for special_state, reward in parameters.get('special_states', []):
    rewards[special_state] = reward
    # print('reward', reward)

# Define state types
terminals = [terminal[0] for terminal in parameters.get('terminals', [])]
special_states = [special_state[0] for special_state in parameters.get('special_states', [])]
forbidden_states = parameters.get('forbidden_states', [])
# print('rewards', rewards)

# Define actions
actions = ['U', 'D', 'L', 'R']

# Define movement directions
directions = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}


# Q-Learning Algorithm
def q_learning(epsilon, num_trials):
    Q = np.zeros((N, M, len(actions)))
    state_action_count = np.zeros((N, M, len(actions)))
    policy = np.full((N, M), None)
    utilities = np.zeros((N, M))
    trial_rewards = []

    for trial in range(num_trials):
        state = parameters['start_state']
        total_reward = 0

        while state not in terminals and state not in forbidden_states:
            if np.random.rand() < epsilon:
                action_index = np.random.choice(len(actions))  # Exploration
            else:
                action_index = np.argmax(Q[state[0], state[1], :])  # Exploitation

            action = actions[action_index]
            next_state, reward = simulate_environment(state, action)
            total_reward += reward

            # Update Q-value
            state_action_count[state[0], state[1], action_index] += 1
            alpha = 1.0 / state_action_count[state[0], state[1], action_index]
            best_next_action = np.argmax(Q[next_state[0], next_state[1], :])
            td_target = reward + gamma * Q[next_state[0], next_state[1], best_next_action]
            td_error = td_target - Q[state[0], state[1], action_index]
            Q[state[0], state[1], action_index] += alpha * td_error

            state = next_state

        trial_rewards.append(total_reward)

    # Derive policy and utilities from Q-values
    for i in range(N):
        for j in range(M):
            if (i, j) not in terminals and (i, j) not in forbidden_states:
                best_action_index = np.argmax(Q[i, j, :])
                policy[i, j] = actions[best_action_index]
                utilities[i, j] = np.max(Q[i, j, :])

    return Q, policy, utilities, trial_rewards


# Simulate environment response to agent action
def simulate_environment(state, action):
    if np.random.rand() < p1:
        selected_action = action
    elif np.random.rand() < p2:
        selected_action = actions[(actions.index(action) + 1) % 4]
    elif np.random.rand() < p3:
        selected_action = actions[(actions.index(action) - 1) % 4]
    else:
        selected_action = actions[(actions.index(action) + 2) % 4]

    next_state = (state[0] + directions[selected_action][0], state[1] + directions[selected_action][1])
    next_state = (min(max(next_state[0], 0), N - 1), min(max(next_state[1], 0), M - 1))

    if next_state in forbidden_states:
        next_state = state  # Stay in the current state if next state is forbidden

    reward = rewards[next_state[0], next_state[1]]
    return next_state, reward

# Run the Q-learning algorithm for user-specified number of trials
Q, policy, utilities, trial_rewards = q_learning(epsilon=epsilon, num_trials=args.iterations)


# Print results
def print_results(policy, utilities, Q):
    print("\nOptimal Policy:")
    for i in range(N):
        for j in range(M):
            if (i, j) in terminals:
                print("| T  ", end="")
            elif (i, j) in forbidden_states:
                print("| F  ", end="")
            else:
                print("| {}  ".format(policy[i, j]), end="")
        print("|")

    print("\nUtility values:")
    for i in range(N):
        for j in range(M):
            if (i, j) in terminals:
                print("| T({:.2f})".format(rewards[i, j]), end="")
            elif (i, j) in forbidden_states:
                print("| F  ", end="")
            else:
                print("| {:.2f} ".format(utilities[i, j]), end="")
        print("|")

    print("\nQ values:")
    for i in range(N):
        for j in range(M):
            print("| ", end="")
            for a in range(len(actions)):
                print("{:.2f} ".format(Q[i, j, a]), end="")
            print("|", end="")
        print()


# Plotting the rewards over trials
# def plot_trial_rewards(trial_rewards):
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(trial_rewards)), trial_rewards)
#     plt.xlabel('Trial')
#     plt.ylabel('Total Reward')
#     plt.title(f'Total Reward over Trials (epsilon={epsilon})')
#     plt.grid(True)
#     plt.savefig(f"trial_rewards_epsilon_{epsilon}.png")
#     plt.show()


class GridWorld:
    def __init__(self, N, M, values, terminals, walls, special_states, actions):
        self.N = N
        self.M = M
        self.values = values
        self.terminals = terminals
        self.walls = walls
        self.special_states = special_states
        self.actions = actions

    def plot_policy_values_and_q_values(self, policy, Q, fig_size=(12, 12), font_size=14, num_trials=0, epsilon=0):
        unit = min(fig_size[1] // self.N, fig_size[0] // self.M)
        unit = max(1, unit)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')

        # Add column numbers
        for i in range(self.M):
            ax.text((i + 0.5) * unit, -0.3 * unit, str(i), ha='center', va='center', fontweight='bold', fontsize=font_size)

        # Add row numbers
        for i in range(self.N):
            ax.text(-0.3 * unit, (self.N - i - 0.5) * unit, str(i), ha='center', va='center', fontweight='bold', fontsize=font_size)

        for i in range(self.M + 1):
            if i == 0 or i == self.M:
                ax.plot([i * unit, i * unit], [0, self.N * unit], color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.N * unit], alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.N + 1):
            if i == 0 or i == self.N:
                ax.plot([0, self.M * unit], [i * unit, i * unit], color='black')
            else:
                ax.plot([0, self.M * unit], [i * unit, i * unit], alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.N):
            for j in range(self.M):
                y = (self.N - 1 - i) * unit
                x = j * unit
                if (i, j) in self.walls:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black', alpha=0.6)
                    ax.add_patch(rect)
                elif (i, j) in self.terminals:
                    color = 'red' if (i, j) == (1, 3) else 'green'
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor=color, alpha=0.6)
                    ax.add_patch(rect)
                    reward = '+100'
                    ax.text(x + 0.5 * unit, y + 0.2 * unit, f'{reward}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=font_size, weight='bold')
                    if policy[i, j] is None:
                        ax.text(x + 0.5 * unit, y + 0.7 * unit, 'End',
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=font_size, weight='bold')
                    else:
                        symbol_map = {'R': '>', 'L': '<', 'U': '^', 'D': 'v'}
                        action_symbol = symbol_map[self.actions[policy[i, j]]]
                        ax.text(x + 0.5 * unit, y + 0.7 * unit, action_symbol,
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=font_size, weight='bold')
                elif (i, j) in self.special_states:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='blue', alpha=0.6)
                    ax.add_patch(rect)
                    max_q_value = np.max(Q[i, j])
                    for idx, action in enumerate(self.actions):
                        color = 'yellow' if Q[i, j, idx] == max_q_value else 'white'
                        ax.text(x + 0.1 * unit, y + (0.2 + 0.2 * idx) * unit, f'| {action}   {Q[i, j, idx]:.4f} |',
                                horizontalalignment='left', verticalalignment='center',
                                fontsize=font_size * 0.7, weight='bold', color=color)
                else:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='white', alpha=0.6)
                    ax.add_patch(rect)
                    max_q_value = np.max(Q[i, j])
                    for idx, action in enumerate(self.actions):
                        color = 'blue' if Q[i, j, idx] == max_q_value else 'black'
                        ax.text(x + 0.1 * unit, y + (0.2 + 0.2 * idx) * unit, f'| {action}   {Q[i, j, idx]:.4f} |',
                                horizontalalignment='left', verticalalignment='center',
                                fontsize=font_size * 0.7, weight='bold', color=color)

        # plt.tight_layout()
        plt.title(f'Q values for ε = {epsilon}, Iterations = {num_trials}')
        plt.show()
        plt.savefig(f"{self.N}x{self.M}_{num_trials}_{epsilon}.png")

# Create GridWorld object
print_results(policy, utilities, Q)
grid_world = GridWorld(N, M, utilities, terminals, forbidden_states, special_states, actions)
# Plot policy, values, and Q-values with specified font size and provided num_trials and epsilon
grid_world.plot_policy_values_and_q_values(policy, Q, font_size=15, num_trials=args.iterations, epsilon=epsilon)

# Execute the printing and plotting functions


