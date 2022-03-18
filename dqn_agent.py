import re
import numpy as np
from environment import Env
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

EPISODES = 1000
STATE_SIZE = 20 #TODO define actual state size

class ReplayBuffer():
    def __init__(self, mem_size, in_features):
        self.mem_counter = 0
        self.mem_size = mem_size
        self.state_memory = np.zeros((self.mem_size, in_features), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, in_features), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_= self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DQN(nn.Module):
    def __init__(self, in_features, n_actions, lr = 0.001):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions))

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

class DQNAgent():
    def __init__(self, gamma, epsilon, in_features, n_actions, batch_size, lr = 0.001, 
            max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, update_tn = 1000):
        self.load_model = False
        self.action_space = [i for i in range(n_actions)]
        self.state_size = in_features
        # Hyperparameters
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.discount_factor = gamma
        self.learning_rate = lr
        self.epsilon = epsilon
        self.epsilon_decay = eps_dec
        self.epsilon_min = eps_end
        self.ls_counter = 0
        self.update_tn_counter = update_tn

        self.memory = ReplayBuffer(max_mem_size, in_features)

        self.Q_eval = DQN(in_features, n_actions)

        self.Q_target = DQN(in_features, n_actions)

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/dqn_trained.h5')

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.Q_eval.device)
        rewards = torch.tensor(reward).to(self.Q_eval.device)
        dones = torch.tensor(done).to(self.Q_eval.device)
        actions = torch.tensor(action).to(self.Q_eval.device)
        states_ = torch.tensor(state_).to(self.Q_eval.device)

        return states, actions, rewards, states_, dones

    def update_target_network(self):
        if self.ls_counter % self.update_tn_counter == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def train(self):
        if self.memory.mem_counter < self.batch_size:
            return
            
        self.Q_eval.optimizer.zero_grad()

        self.update_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_eval = self.Q_eval.forward(states)[indices, actions.type(torch.long)]
        q_target = self.Q_target.forward(states_).max(dim=1)[0]

        q_target[dones] = 0.0
        q_value = rewards + self.discount_factor * q_target

        loss = self.Q_eval.loss(q_value, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.ls_counter += 1

        self.decrement_epsilon()

if __name__ == "__main__":
    env = Env()
    agent = DQNAgent(0.99, 1.0, env.observation_space.shape[0], env.action_size, 32)

    best_score = -np.inf

    global_step = 0
    scores, eps_history, step = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        # state = np.reshape(state, [1, STATE_SIZE])

        while not done:
            global_step += 1

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            # next_state = np.reshape(state, [1, STATE_SIZE])
            
            score += reward

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state

        scores.append(score)
        step.append(global_step)

        avg_score = np.mean(scores[-100:])

        print('episode: ', e,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', global_step)

        if avg_score > best_score:
            best_score = avg_score

        eps_history.append(agent.epsilon)


