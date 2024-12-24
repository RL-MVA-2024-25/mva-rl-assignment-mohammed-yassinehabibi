import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gymnasium as gym
from statistics import mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor network
class Actor(nn.Module): 
     def __init__(self, in_size, out_size): 
          super(Actor, self).__init__()
          self.linear1 = nn.Linear(in_size, 128)
          self.linear2 = nn.Linear(128, 128)
          self.linear3 = nn.Linear(128, out_size)
          #self.dropout = nn.Dropout(0.7)
          self.softmax = nn.Softmax(dim=1)

          self.policy_history = Variable(torch.Tensor()).to(device)
          self.reward_episode = []

          self.reward_history = []
          self.loss_history = []
  
     def forward(self, x): 
          # Convert numpy state to tensor
          x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
          x = F.relu(self.linear1(x))
          x = F.relu(self.linear2(x))
          #x = self.dropout(x)
          x = self.softmax(self.linear3(x))
          return x

# Critic network
class Critic(nn.Module): 
     def __init__(self, in_size): 
          super(Critic, self).__init__()
          self.linear1 = nn.Linear(in_size, 128)
          self.linear2 = nn.Linear(128, 128)
          self.linear3 = nn.Linear(128, 1)
          #self.dropout = nn.Dropout(0.7)

          self.value_episode = []
          self.value_history = Variable(torch.Tensor()).to(device)
     
     def forward(self, x): 
          x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
          x = F.relu(self.linear1(x))
          x = F.relu(self.linear2(x))
          x = self.linear3(x)
          return x 

# Combined module (mostly for loading / storing)
class ActorCritic(nn.Module): 
     def __init__(self, actor, critic): 
          super(ActorCritic, self).__init__()
          self.actor = actor
          self.critic = critic
  
     def forward(self, x):
          value = self.critic(x)
          policy = self.actor(x)
          return value, policy

def evaluate_actor(agent: Actor, env: gym.Env, nb_episode: int = 10, scaling='standard') -> float:
    """
    Evaluate an actor agent in a given environment.

    Args:
        agent (Actor): The actor agent to evaluate.
        env (gym.Env): The environment to evaluate the agent in.
        nb_episode (int): The number of episodes to evaluate the agent.

    Returns:
        float: The mean reward of the agent over the episodes.
    """
    rewards: list[float] = []
    states_means = np.load('states_means.npy')
    states_stds = np.load('states_stds.npy')
    for _ in range(nb_episode):
        obs, info = env.reset()
        if scaling == 'standard':
            obs = (obs - states_means) / states_stds
        elif scaling == 'log':
            obs = np.log(obs + 1)
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action = agent(obs)
            action = torch.argmax(action).item()
            obs, reward, done, truncated, _ = env.step(action)
            if scaling == 'standard':
                obs = (obs - states_means) / states_stds
            elif scaling == 'log':
                obs = np.log(obs + 1)
            episode_reward += reward
        rewards.append(episode_reward)
    return mean(rewards)