import torch
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from dqn_agent import dqn_agent
from dqn_greedy_action import greedy_action
from dqn_ import DQN
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
          'learning_rate': 0.002,
          'gamma': 0.85,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 25000,
          'epsilon_delay_decay':20,
          'batch_size': 64,
          'gradient_steps': 3,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 50,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 50,
          'neurons': 256,
          'max_episode': 750}
        DQN = torch.nn.Sequential(nn.Linear(6, self.config['neurons']),
                          nn.ReLU(),
                          nn.Linear(self.config['neurons'], self.config['neurons']),
                          nn.ReLU(), 
                          nn.Linear(self.config['neurons'], self.config['nb_actions']).to(device)
        )
        self.agent = dqn_agent(self.config, DQN)

    def act(self, observation, use_random=False):
        return greedy_action(self.agent.model, observation)

    def save(self, path):
        torch.save(self.agent.model.state_dict(), path)

        pass

    def train(self):
        print('training')
        self.agent.train(env, self.config['max_episode'])
                         

    def load(self) -> None:

        path = 'saves'
        model = DQN = torch.nn.Sequential(nn.Linear(6, self.config['neurons']),
                          nn.ReLU(),
                          nn.Linear(self.config['neurons'], self.config['neurons']),
                          nn.ReLU(), 
                          nn.Linear(self.config['neurons'], self.config['nb_actions']).to("cpu")
        )

        state_dict = torch.load('saves')
        model.load_state_dict(state_dict)
        self.agent = dqn_agent(self.config,model)

