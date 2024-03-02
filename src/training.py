import random
import os
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent  # Replace DummyAgent with your agent implementation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


seed_everything(seed=42)
# Initialization of the agent. Replace DummyAgent with your custom agent implementation.
agent = ProjectAgent()

agent.train()
agent.save('saves')