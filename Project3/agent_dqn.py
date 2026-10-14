#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import math
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # neural net initialization
        self.args = args
        self.env = env
        self.num_actions = env.action_space.n

        self.in_channels = 4
        self.q_net = DQN(self.in_channels, self.num_actions).to(self.device)
        self.target_q_net = DQN(self.in_channels, self.num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.model_path = 'model/dqn_model.pth'

        self.lr = 0.0001
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.batch_size = 32

        self.replay_buffer_size = 10000
        self.buffer = deque(maxlen=self.replay_buffer_size)
        

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.train_frequency = 4
        self.learning_starts = 4 * self.batch_size

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.q_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.q_net.eval()
            ###########################
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        wrapped_env = getattr(self.env, 'env', self.env)
        action_meanings = None

        if hasattr(wrapped_env, 'unwrapped') and hasattr(wrapped_env.unwrapped, 'get_action_meanings'):
            action_meanings = wrapped_env.unwrapped.get_action_meanings()
        elif hasattr(wrapped_env, 'get_action_meanings'):
            action_meanings = wrapped_env.get_action_meanings()

        if action_meanings is not None and 'FIRE' in action_meanings:
            # Breakout starts on an intro screen; issue the FIRE action once to begin a life.
            fire_action = action_meanings.index('FIRE')
            fire_obs, _, _, _, _ = wrapped_env.step(fire_action)
            return fire_obs
        ###########################
        return None
    
    
    def _format_state(self, observation, add_batch_dim=False):
        arr = np.asarray(observation, dtype=np.float32)
        if arr.ndim == 3:  # (H, W, C) -> (C, H, W)
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim == 4 and arr.shape[-1] == self.in_channels:  # (N, H, W, C) -> (N, C, H, W)
            arr = np.moveaxis(arr, -1, 1)

        tensor = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        tensor = tensor.contiguous()
        if add_batch_dim and tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _batch_states_to_device(self, observations):
        """Convert one replay batch to NCHW and transfer it in one operation."""
        states = np.asarray(observations, dtype=np.uint8)
        states = np.moveaxis(states, -1, 1)
        states = np.ascontiguousarray(states, dtype=np.float32)
        return torch.from_numpy(states).to(self.device)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if not test and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        state = self._format_state(observation, add_batch_dim=True)
        with torch.no_grad():
            q_values = self.q_net(state)
            action = torch.argmax(q_values, dim=1).item()  # Exploit: select the action with max Q-value
        
        ###########################
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        self.buffer.append((
            np.asarray(state, dtype=np.uint8),
            action,
            reward,
            np.asarray(next_state, dtype=np.uint8),
            done,
        ))
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        buffer_batch = random.sample(self.buffer, self.batch_size)
        # states, actions, rewards, next_states, dones = zip(*buffer_batch)
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
  
        ###########################
        return buffer_batch

    # def explore_env(self, n_episodes=1000):
    #     """
    #     collect data from the enviroment. Essentially, get episodes and store then. An episode ends when the game is over.

    #     Each episode is list of tuples (state, action, reward, next_state, done). Each tuple is a step in the episode.

    #     The next episode is a new list and starts when the game is reset. The game is reset when the agent loses all its lives or when the game is won. 
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
    #     episodes = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state, done)
    #     time_steps = [] # time_steps in each episode, each time step is a tuple (state, action, reward, next_state, done) 

    #     state = self.env.reset()
    #     for _ in range(n_episodes):
    #         done = False
    #         while not done:
    #             action = self.make_action(state, test=True)
    #             next_state, reward, done, truncated, _ = self.env.step(action)
    #             done = done or truncated
    #             time_steps.append((state, action, reward, next_state, done))
    #             episodes.append(time_steps)
    #             state = next_state
    #         state = self.env.reset()  # Reset the environment for the next episode
    #         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Decay epsilon after each episode
        
        ###########################

    def plot_training_curve(self, avg_reward, time_steps):
        """
        Plot the training curve of average reward vs time steps.
        """
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, avg_reward)
        plt.xlabel('Time Steps')
        plt.ylabel('Average Reward (over last 30 lives)')
        plt.title('Training Curve')
        plt.grid()
        plt.savefig('training_curve.png')
        # plt.show()

    def train(self, n_episodes=5000):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        print(f"Training DQN agent using device: {self.device}...")
        lives_in_episode = 5  # 5 lives per episode

        model_save_threshold = 0  # Save model if average reward over last 30 lives is greater than this threshold

        # overall_loss = math.inf
        recent_episode_returns = deque(maxlen=30)
        avg_reward = []

        update_steps = 0
        time_step = 0
        # explore the environment and collect data
        state = self.env.reset()
        start_state = self.init_game_setting()
        if start_state is not None:
            state = start_state
        for i in tqdm.tqdm(range(n_episodes), desc=f"Training DQN Agent"):
            # print(f"Episode {i + 1}/{n_episodes}")
            done = False
            episode_return = 0.0
            while lives_in_episode > 0:
                action = self.make_action(state, test=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_return += float(reward)

                if done:
                    lives_in_episode -= 1

                    # Keep only the most recent 30 life returns. The deque drops the
                    # oldest value automatically once it reaches maxlen=30.
                    recent_episode_returns.append(episode_return)

                    # Average reward over the last 30 lives, not per step.
                    if len(recent_episode_returns) == 30:
                        avg_reward.append(np.mean(recent_episode_returns))
                    episode_return = 0.0

                self.push(state, action, reward, next_state, done)
                state = next_state

                time_step += 1
                if len(self.buffer) >= self.learning_starts and time_step % self.train_frequency == 0:
                    buffer_batch = self.replay_buffer()
                    states, actions, rewards_batch, next_states, dones = zip(*buffer_batch)
                    states = self._batch_states_to_device(np.stack(states, axis=0))
                    next_states = self._batch_states_to_device(np.stack(next_states, axis=0))
                    actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
                    rewards_batch = torch.FloatTensor(rewards_batch).unsqueeze(1).to(self.device)
                    dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

                    # Compute target Q-values using the target network
                    with torch.no_grad():
                        target_q_values = self.target_q_net(next_states)
                        max_target_q_values, _ = torch.max(target_q_values, dim=1, keepdim=True)
                        targets = rewards_batch + (self.gamma * max_target_q_values * (1 - dones))

                    # Compute current Q-values using the main network
                    current_q_values = self.q_net(states).gather(1, actions)

                    # Compute loss
                    loss = self.loss_fn(current_q_values, targets)

                    # Backpropagation and optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # Gradient clipping
                    self.optimizer.step()

                    update_steps += 1

                    # Update target network periodically
                    if update_steps % 500 == 0:
                        self.target_q_net.load_state_dict(self.q_net.state_dict())

                        if avg_reward and avg_reward[-1] > model_save_threshold:
                            model_save_threshold = avg_reward[-1]
                            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                            torch.save(self.q_net.state_dict(), self.model_path)
                            print(f"Model saved at time step: {time_step} with average reward: {model_save_threshold}")

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Reset the environment and lives for the next episode
            state = self.env.reset()
            start_state = self.init_game_setting()
            if start_state is not None:
                state = start_state
            lives_in_episode = 5

        # Plot the training curve after training
        time_steps = list(range(len(avg_reward)))
        self.plot_training_curve(avg_reward, time_steps)

        
        ###########################
