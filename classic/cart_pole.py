""" CartPole enviroment test
"""
import gym


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    episode_count = 100
    env = gym.make('CartPole-v0')
    agent = RandomAgent(env.action_space)

    for i in range(episode_count):
        ob = env.reset()
        env.render()
        reward = 0
        total_reward = 0
        done = False
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
        print('Episode: {}, reward={}'.format(i+1, total_reward))

    env.close()
