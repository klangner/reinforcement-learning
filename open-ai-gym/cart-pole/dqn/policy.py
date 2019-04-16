import numpy as np


class EpsilonGreedyPolicy:

    def __init__(self, network, epsilon_min = 0.01, epsilon_max = 0.01, epsilon_lambda = 0.001):
        self.network = network
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_lambda = epsilon_lambda
        self.epsilon = self.epsilon_max        
        self.step = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, network.output_shape)
        else:
            qvalues = self.model.predict(state)
            action = np.argmax(qvalues)
            self.step += 1
        self.epsilon = self.epsilon_min + \
            (self.epsilon_max-self.epsilon_min) * np.power(np.e, self.epsilon_lambda*self.step)
        return action


class GreedyPolicy:
    """Good policy for evaluating the model
    """

    def __init__(self, network):
        self.network = network

    def select_action(self, state):
        qvalues = self.model.predict(state)
        action = np.argmax(qvalues)
        return action