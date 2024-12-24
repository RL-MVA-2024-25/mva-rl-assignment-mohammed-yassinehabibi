from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
from torch import nn
from dqn_agent import dqn_agent, greedy_action
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
# Declare network
class ProjectAgent:
    def __init__(self):
        self.state_dim = env.observation_space.shape[0]
        self.n_action = env.action_space.n 
        self.nb_neurons=128
        self.model = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.n_action)).to(device)

        # DQN config
        self.model_path= 'best_model_standard_scaling.pth'
        self.scaling='standard'
        self.states_means = np.load('states_means.npy')
        self.states_stds = np.load('states_stds.npy')

        pass


    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        if self.scaling == 'log':
            observation = np.log10(observation + 1)
        elif self.scaling == 'standard':
            observation = (observation - self.states_means) / self.states_stds
        return greedy_action(self.model, observation)

    def save(self, path):
        pass

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False))
        pass
