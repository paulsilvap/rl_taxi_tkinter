import numpy as np
from environment import Env

EPISODES = 1000
STATE_SIZE = 20 #TODO define actual state size

class DQNAgent:
    def __init__(self):
        self.load_model = False
        self.action_space = [0,1,2,3,4]
        self.action_size = len(self.action_space)
        self.state_size = STATE_SIZE 
        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/dqn_trained.h5')

    def build_model(self):
        pass

    def get_action(self, state):
        pass

    def train_model(self, state, action, reward, next_state, next_action, done):
        pass

if __name__ == "__main__":
    env = Env()
    agent = DQNAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        # state = np.reshape(state, [1, STATE_SIZE])

        while not done:
            global_step += 1

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            # next_state = np.reshape(state, [1, STATE_SIZE])
            next_action = agent.get_action(next_state) 

