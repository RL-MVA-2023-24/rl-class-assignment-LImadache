import torch
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from dqn_agent import dqn_agent
from dqn_greedy_action import greedy_action
import torch.nn as nn
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        device = "cpu"
        self.config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000,
          'epsilon_delay_decay': 20,
          'batch_size': 20,
          'gradient_steps': 10,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 500,
          'update_target_tau': 0.001,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 50,
          'neurons': 1024}
        DQN = torch.nn.Sequential(nn.Linear(6, self.config['neurons']),
                          nn.ReLU(),
                          nn.Linear(self.config['neurons'], self.config['neurons']),
                          nn.ReLU(), 
                          nn.Linear(self.config['neurons'], self.config['neurons']),
                          nn.ReLU(), 
                          nn.Linear(self.config['neurons'], self.config['neurons']),
                          nn.ReLU(), 
                          nn.Linear(self.config['neurons'], 4)).to(device)
        self.agent = dqn_agent(self.config, DQN)

    def act(self, observation, use_random=False):
        return greedy_action(self.agent.model, observation)

    def save(self, path):
        torch.save(self.dqn_agent.model.state_dict, path)

        pass

    def train(self, max_episode = 20000):
        self.agent.train(env, max_episode)
                         

    def load(self) -> None:
        ### SAVE IN THE REPOSITORY
        path = 'saves'
        if not os.path.exists(path):
            print("No model to load")
            return
        with open(path, 'rb') as f:
            self.model = torch.load(f)
            print("model loaded")
