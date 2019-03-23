
class Agent:

    def collect_policy(self, state):
        pass

    def evaluate_policy(self, state):
        pass


def train_on_episode(env, agent, time_max=1000):
    """Collect experience by running single episode

    Params:
        env: OpenAI gym like interface
        agent: Agent class
        time_max: Maximum number of steps

    Return:
        total reward received
    """
    
    states,actions = [],[]
    total_reward = 0
    
    state = env.reset()
    
    for t in range(time_max):
        action = agent.collect_policy(state)
        next_state, reward, done, info = env.step(action)
        agent.add_observation(state, action, reward, next_state, done)
        agent.train()
        total_reward += reward
        state = next_state
        if done: break
            
    return total_reward


def evaluate_episode(env, agent, time_max=1000):
    """Evaluate agent on single episode

    Params:
        env: OpenAI gym like interface
        agent: Agent class
        time_max: Maximum number of steps

    Return:
        total reward received
    """
    
    states,actions = [],[]
    total_reward = 0
    
    state = env.reset()
    
    for t in range(time_max):
        action = agent.evaluate_policy(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        if done: break
            
    return total_reward