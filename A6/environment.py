import numpy as np
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class GridWorld:
    def __init__(self, size=5, traps=None, goal=None, obstacles=None, random_obstacles=False, num_random_obstacles=3):
        self.size = size
        self.rewards = np.zeros((size, size))
        self.is_terminal_state = np.full((size, size), False, dtype=bool)
        self.actions = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.actions)
        self.action_prob = 1.0 / self.num_actions  # For a random policy

        # Define goal state
        self.goal_pos = goal if goal is not None else (size - 1, size - 1)
        self.rewards[self.goal_pos] = 10
        self.is_terminal_state[self.goal_pos] = True

        # Define trap states
        if traps is None:
            self.traps_pos = [(1, 1), (2, 3), (3, 0)]  # Default trap positions
        else:
            self.traps_pos = traps

        for trap in self.traps_pos:
            self.rewards[trap] = -10
            self.is_terminal_state[trap] = True

        # Define obstacles
        self.obstacles_pos = []
        if random_obstacles:
            self.obstacles_pos = self._generate_random_obstacles(num_random_obstacles)
        elif obstacles is not None:
            self.obstacles_pos = obstacles
        # Example fixed obstacles if not random and none provided
        # elif obstacles is None and not random_obstacles:
        #     self.obstacles_pos = [(0,1), (1,3)]

        # Ensure obstacles are not on goal or traps
        for obs in self.obstacles_pos:
            if obs == self.goal_pos or obs in self.traps_pos:
                # This is a simple handling, ideally, regeneration or more robust placement is needed
                print(f"Warning: Obstacle at {obs} coincides with goal or trap. It will be ignored.")
            elif self.is_valid_pos(obs[0], obs[1]):
                self.rewards[obs] = 0  # Obstacles are like walls, no reward, agent cannot move there
                # Agents should not be able to enter obstacle cells. This will be handled in `get_next_state`

        self.current_pos = (0, 0)  # Starting position

    def _generate_random_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                obs_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if obs_pos != self.goal_pos and obs_pos not in self.traps_pos and obs_pos not in obstacles and obs_pos != (
                        0, 0):  # Not start
                    obstacles.append(obs_pos)
                    break
        return obstacles

    def is_valid_pos(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def is_obstacle(self, r, c):
        return (r, c) in self.obstacles_pos

    def reset(self):
        """Resets the agent to a random start state or a fixed one."""
        # For this project, let's use a fixed start state for consistency in policy evaluation
        self.current_pos = (0, 0)
        return self.current_pos

    def step(self, action_index):
        """
        Takes an action and returns the next state, reward, and done status.
        Uses a stochastic policy where the chosen action is taken with some probability,
        and other actions can be taken with lower probabilities (not implemented here for simplicity,
        as the project asks for a random policy where each action has 0.25 prob).
        For the random policy evaluation, the agent *selects* actions randomly.
        The environment's transition is deterministic given an action.
        """
        action = self.actions[action_index]
        r, c = self.current_pos

        next_r, next_c = r, c

        if action == 'up':
            next_r = max(0, r - 1)
        elif action == 'down':
            next_r = min(self.size - 1, r + 1)
        elif action == 'left':
            next_c = max(0, c - 1)
        elif action == 'right':
            next_c = min(self.size - 1, c + 1)

        # Check if next state is an obstacle
        if self.is_obstacle(next_r, next_c):
            next_r, next_c = r, c

        self.current_pos = (next_r, next_c)
        reward = self.rewards[next_r, next_c]
        done = self.is_terminal_state[next_r, next_c]

        return self.current_pos, reward, done

    def get_state_index(self, state_pos=None):
        """Converts (row, col) position to a single state index."""
        if state_pos is None:
            state_pos = self.current_pos
        return state_pos[0] * self.size + state_pos[1]

    def get_pos_from_index(self, state_index):
        """Converts a single state index back to (row, col) position."""
        r = state_index // self.size
        c = state_index % self.size
        return r, c

    def get_all_states(self):
        """Returns a list of all possible (row, col) states, excluding obstacles."""
        states = []
        for r in range(self.size):
            for c in range(self.size):
                if not self.is_obstacle(r, c):
                    states.append((r, c))
        return states

    def render(self):
        """Prints the grid to the console."""
        grid_repr = np.full((self.size, self.size), '_', dtype=str)
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.goal_pos:
                    grid_repr[r, c] = 'G'
                elif (r, c) in self.traps_pos:
                    grid_repr[r, c] = 'X'
                elif (r, c) in self.obstacles_pos:
                    grid_repr[r, c] = '#'
                if (r, c) == self.current_pos:
                    grid_repr[r, c] = 'A'  # Agent

        for row in grid_repr:
            print(' '.join(row))
        print("-" * (self.size * 2 - 1))


# Example Usage (can be moved to a main script or notebook)
if __name__ == '__main__':
    # Define custom positions
    custom_traps = [(1, 2), (3, 3)]
    custom_goal = (4, 4)
    custom_obstacles = [(0, 1), (1, 0), (2, 2)]

    # env = GridWorld(size=5, traps=custom_traps, goal=custom_goal, obstacles=custom_obstacles)
    env = GridWorld(size=5)  # Using default traps, goal, and no obstacles initially
    # env = GridWorld(size=5, random_obstacles=True, num_random_obstacles=4)

    print("Initial Grid:")
    env.render()
    print(f"Rewards:\n{env.rewards}")
    print(f"Terminal States:\n{env.is_terminal_state}")
    print(f"Obstacles: {env.obstacles_pos}")
    print(f"All non-obstacle states: {env.get_all_states()}")
    print(f"Number of actions: {env.num_actions}")

    # Test random policy interaction
    print("\nSimulating one episode with random policy:")
    current_state = env.reset()
    env.render()
    done = False
    total_reward = 0
    episode_length = 0
    while not done:
        action = random.choice(range(env.num_actions))
        next_state, reward, done = env.step(action)
        print(
            f"Current: {env.get_pos_from_index(env.get_state_index(current_state))}, Action: {env.actions[action]}, Next: {next_state}, Reward: {reward}, Done: {done}")
        env.render()
        current_state = next_state
        total_reward += reward
        episode_length += 1
        if episode_length > 100:
            print("Episode too long, breaking.")
            break
    print(f"Episode finished. Total reward: {total_reward}, Length: {episode_length}")
