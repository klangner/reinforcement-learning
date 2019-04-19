import numpy as np
import time
import gym
from dqn.policy import RandomPolicy


NUM_EPISODES = 10_000
MAX_STEPS = 1_000

# Take inner environment so we are not restricted to 200 steps
env = gym.make("CartPole-v0").env
observation_shape = env.observation_space.shape
n_actions = env.action_space.n

random_policy = RandomPolicy(n_actions)
start_time = time.time()
score = []
for episode in range(NUM_EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        action = random_policy.select_action(state)
        next_state, reward, done, info = env.step(action)
        if done:
            score.append(step)
            break

total_time = time.time() - start_time
mean_score = np.mean(score)
print('{} episoded finished in {:.2f} sec. Avg reward: {:.2f}'.format(NUM_EPISODES, total_time, mean_score))