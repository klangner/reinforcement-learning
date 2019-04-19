import numpy as np
import matplotlib.pyplot as plt
import gym


env = gym.make("CartPole-v0") 

observation_shape = env.observation_space.shape
n_actions = env.action_space.n

print("Observation shape: {}".format(observation_shape))
print("Number of actions: {}".format(n_actions))

print("Example state: {}".format(env.reset()))
plt.imshow(env.render('rgb_array'))

input("Press any key...")