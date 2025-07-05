import gym 
import gym_anytrading
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import pandas as pd
from gym_anytrading.envs import StocksEnv
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from neural_network import TradingQNetwork
from collections import deque
import numpy as np
import random



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = TradingQNetwork(state_size, action_size).to(device)
        self.target_model = TradingQNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_network()
        self.episodes_done = 0
        self.max_episodes = 10000

    

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)

        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = self.process_minibatch(minibatch)

        q_values = self.model(states)
        q_selected= q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values, _ = next_q_values.max(dim=1)

        targets = rewards + (1 - dones.float()) * self.gamma * max_next_q_values
        
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def epsilon_greedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
            
    def process_minibatch(self, minibatch):
        states = torch.FloatTensor(np.array([s[0].flatten() for s in minibatch])).to(device)
        actions = torch.LongTensor([s[1] for s in minibatch]).to(device)
        rewards = torch.FloatTensor([s[2] for s in minibatch]).to(device)
        next_states = torch.FloatTensor(np.array([s[3].flatten() for s in minibatch])).to(device)
        dones = torch.BoolTensor([s[4] for s in minibatch]).to(device)
        return states, actions, rewards, next_states, dones

    def load(self, name):
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.update_target_network()
        self.epsilon = checkpoint['epsilon']

        print(f"Epsilon restored: {self.epsilon:.4f}")


    def save(self, name, epsilon):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': epsilon
        }, name)

    def keep_training(self):
        return self.epsilon > self.epsilon_min



