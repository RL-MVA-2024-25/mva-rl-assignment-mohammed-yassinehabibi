import torch
import numpy as np
import random
from statistics import mean
from interface import Agent
import gymnasium as gym


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

def evaluate_agent_2(agent: Agent, env: gym.Env, nb_episode: int = 10, scaling='log') -> float:
    """
    Evaluate an agent in a given environment.

    Args:
        agent (Agent): The agent to evaluate.
        env (gym.Env): The environment to evaluate the agent in.
        nb_episode (int): The number of episode to evaluate the agent.

    Returns:
        float: The mean reward of the agent over the episodes.
    """
    states_means = np.load('states_means.npy')
    states_stds = np.load('states_stds.npy')
    rewards: list[float] = []
    for _ in range(nb_episode):
        obs, info = env.reset()
        env.env.k1, env.env.k2, env.env.f = agent.env_params[np.random.randint(0, len(agent.env_params))]
        if scaling == 'log':
            obs = np.log10(obs + 1)
        elif scaling == 'standard':
            obs = (obs - states_means) / states_stds
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            if scaling == 'log':
                obs = np.log10(obs + 1)
            elif scaling == 'standard':
                obs = (obs - states_means) / states_stds
            episode_reward += reward
        rewards.append(episode_reward)
    env.reset()
    return mean(rewards)    

class dqn_agent_2:
    def __init__(self, config, model, load=False):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = model
        self.criterion = config['criterion'] #torch.nn.SmoothL1Loss() #Or torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.device = device  # Store device
        self.load = load
        self.losses = []
        self.best_model_path = config['model_path']  # File name for saving the best model
        if config['scaling']=='standard':
            self.states_means = np.load('states_means.npy')
            self.states_stds =np.load('states_stds.npy')
            self.scaling = 'standard'
        elif config['scaling']=='log':
            self.scaling = 'log'
        self.env_params = np.load('params.npy')


    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        env.env.k1, env.env.k2, env.env.f = self.env_params[np.random.randint(0, len(self.env_params))]
        if self.scaling == 'log':
            state = np.log10(state + 1)
        elif self.scaling == 'standard':
            state = (state - self.states_means) / self.states_stds
        epsilon = self.epsilon_max
        step = 0

        # Track the best reward
        best_deterministic_reward = -float('inf')
        if self.load:
            best_deterministic_reward = torch.load("best_deterministic_reward_2.pth", weights_only=False)
            print(f"Loaded model with deterministic reward {best_deterministic_reward:.2f}M")
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            true_reward = reward/1_000_000
            reward = np.log10(reward + 100_000)
            if self.scaling == 'log':
                next_state = np.log10(next_state + 1)
            elif self.scaling == 'standard':
                next_state = (next_state - self.states_means) / self.states_stds
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += true_reward

            if step > self.epsilon_delay and step % 1 == 0:
                # train
                self.gradient_step()

            # next transition
            step += 1
            if done or trunc:
                if step > self.epsilon_delay:
                    # Check if current episode return is better than the best
                    deterministic_reward = evaluate_agent_2(self, env, 4, scaling=self.scaling)/1_000_000
                    # Save the best model
                    if deterministic_reward > best_deterministic_reward:
                        best_deterministic_reward = deterministic_reward
                        torch.save(self.model.state_dict(), self.best_model_path)
                        torch.save(best_deterministic_reward, "best_deterministic_reward_2.pth")
                        print("Evaluation deterministic reward", f"{deterministic_reward:.1f}M")
                    #self.gradient_step()
                    print("Episode {:3d}".format(episode), 
                        "epsilon {:6.2f}".format(epsilon), 
                        "batch size {:5d}".format(len(self.memory)), 
                        "episode return {:4.1f}M".format(episode_cum_reward),
                            "loss {:4.1f}".format(self.losses[-1]),
                        sep=', ')
                episode += 1
                state, _ = env.reset()
                env.env.k1, env.env.k2, env.env.f = self.env_params[np.random.randint(0, len(self.env_params))]
                if self.scaling == 'log':
                    state = np.log10(state + 1)
                elif self.scaling == 'standard':
                    state = (state - self.states_means) / self.states_stds
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return

    def act(self, state):
        return greedy_action(self.model, state)