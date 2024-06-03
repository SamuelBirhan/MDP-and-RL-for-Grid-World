import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


# Read grid world parameters from text file
# Define the command-line arguments
parser = argparse.ArgumentParser(description='Run value iteration on a grid world.')
parser.add_argument('datafile', type=str, help='Path to the data file containing grid world parameters')

# Parse the command-line arguments
args = parser.parse_args()

# Read grid world parameters from the specified data file
file_path = args.datafile
parameters = read_parameters_from_file(file_path)

# Extracting parameters from the dictionary
N = parameters['N']
M = parameters['M']
start_state = parameters['start_state']
p1 = parameters['p1']
p2 = parameters['p2']
p3 = parameters['p3']
r = parameters['r']
gamma = parameters.get('gamma', 0.90)
# print(gamma)
# epsilon = parameters.get('epsilon', 0.0001)

# Initialize rewards matrix with default reward value
rewards = np.full((N, M), r)

# Assign rewards for terminal states
for terminal, reward in parameters.get('terminals', []):
    row, col = terminal
    rewards[row, col] = reward

# Assign rewards for special states
for special_state, reward in parameters.get('special_states', []):
    row, col = special_state
    rewards[row, col] = reward

# # Print rewards for verification
# print("Rewards:")
# for row in rewards:
#     print(row)

# Define state types
terminals = [terminal[0] for terminal in parameters.get('terminals', [])]
walls = parameters.get('forbidden_states', [])
special_states = [special_state[0] for special_state in parameters.get('special_states', [])]

# Define actions
actions = ['U', 'D', 'L', 'R']

# Define movement directions
directions = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}


# Value Iteration Algorithm
def value_iteration():
    utilities = np.zeros((N, M))  # Initialize utilities with zeros
    policy = np.full((N, M), None)
    delta = float('inf')
    utilities_list = []
    utilities_list.append(np.copy(utilities))

    while delta > 0.001:
        delta = 0
        new_utilities = np.copy(utilities)

        for i in range(N):
            for j in range(M):
                if (i, j) in terminals or (i, j) in walls:
                    continue

                max_utility = float('-inf')
                best_action = None

                # Compute the expected utility for each action
                for action in actions:
                    expected_utility = 0
                    for prob, next_action in [(p1, action), (p2, actions[(actions.index(action) + 1) % 4]),
                                              (p3, actions[(actions.index(action) - 1) % 4]),
                                              (1 - p1 - p2 - p3, actions[(actions.index(action) + 2) % 4])]:
                        next_state = (i + directions[next_action][0], j + directions[next_action][1])

                        next_state = (min(max(next_state[0], 0), N - 1), min(max(next_state[1], 0), M - 1))
                        expected_utility += prob * (rewards[next_state[0], next_state[1]] + gamma * utilities[
                            next_state[0], next_state[1]])
                    if expected_utility > max_utility:
                        max_utility = expected_utility
                        best_action = action

                # Update the utility and policy
                new_utilities[i, j] = max_utility
                policy[i, j] = best_action

                delta = max(delta, abs(new_utilities[i, j] - utilities[i, j]))

        utilities = new_utilities
        utilities_list.append(np.copy(utilities))

    return utilities, policy, utilities_list

# Run the value iteration algorithm
optimal_utilities, optimal_policy, utilities_list = value_iteration()
def plot_convergence(utilities_list):
    plt.figure(figsize=(10, 6))
    for i in range(N):
        for j in range(M):
            if (i, j) not in walls and (i, j) not in terminals:
                plt.plot(range(len(utilities_list)), [utilities[i, j] for utilities in utilities_list],
                         label=f"({i},{j})")
    plt.xlabel('Iterations')
    plt.ylabel('Utility Value')
    plt.title('Utility Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

# Print utilities in grid form
print("\nUtility values:")
for i in range(N):
    for j in range(M):
        if (i, j) in walls:
            print("|", "Wall  ", end="")
        elif (i, j) in terminals:
            print("| T({:+.0f})".format(rewards[i, j]), end="")
        else:
            print("| {:.4f} ".format(optimal_utilities[i, j]), end="")
    print("|")

# Print the optimal policy in grid form
print("\nOptimal Policy:")
for i in range(N):
    for j in range(M):
        if (i, j) in walls:
            print("|", "W ", end="")
        elif (i, j) in terminals:
            print("| T ", end="")
        else:
            print("| {} ".format(optimal_policy[i, j]), end="")
    print("|")

# Plot the utility convergence
plot_convergence(utilities_list)

# Define the GridWorld class
class GridWorld:
    def __init__(self, N, M, values, terminals, walls, special_states):
        self.N = N
        self.M = M
        self.values = values
        self.terminals = terminals
        self.walls = walls
        self.special_states = special_states

    def plot_policy_and_values(self, policy, fig_size=(8, 6), font_size=12):
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
                    reward = rewards[i, j]
                    ax.text(x + 0.5 * unit, y + 0.2 * unit, f'{reward:+.0f}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=font_size, weight='bold')
                    if policy[i, j] is None:
                        ax.text(x + 0.5 * unit, y + 0.7 * unit, 'End',
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=font_size, weight='bold')
                    else:
                        ax.text(x + 0.5 * unit, y + 0.7 * unit, f'{self.values[i, j]:.4f}',
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=font_size, weight='bold')
                elif (i, j) in self.special_states:  # Add condition for special states
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='blue', alpha=0.6)
                    ax.add_patch(rect)
                    ax.text(x + 0.5 * unit, y + 0.2 * unit, f'{self.values[i, j]:.4f}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=font_size, weight='bold')
                else:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='white', alpha=0.6)
                    ax.add_patch(rect)
                    ax.text(x + 0.5 * unit, y + 0.2 * unit, f'{self.values[i, j]:.4f}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=font_size, weight='bold')
                if (i, j) not in self.walls and (i, j) not in self.terminals:
                    a = policy[i, j]
                    symbol = {'U': '^', 'R': '>', 'D': 'v', 'L': '<'}
                    ax.plot([x + 0.5 * unit], [y + 0.8 * unit], marker=symbol[a], linestyle='none',
                            markersize=max(fig_size) * unit, color='lightblue')

        # plt.tight_layout()
        plt.title('Policy and Values')
        plt.show()


# Create GridWorld object
grid_world = GridWorld(N, M, optimal_utilities, terminals, walls, special_states)

# Plot policy and values with specified font size
grid_world.plot_policy_and_values(optimal_policy, font_size=13)
