import numpy as np
import torch
import math 
import random

## could do a dual dropout with Dynamix dropout until it stablises followed by simulateda annealing 
class DynamicDropout:
    def __init__(self):
        self.base_dropout_rate = 0.5
        self.dropout_rate = self.base_dropout_rate
        self.prev_episode_mean_reward = None
        self.reward_difference = 0
    
    def update_dropout_rate(self, episode_rewards):
        current_episode_mean_reward = np.mean(episode_rewards)
        if self.prev_episode_mean_reward is not None:
            self.reward_difference = current_episode_mean_reward - self.prev_episode_mean_reward
            
        self.prev_episode_mean_reward = current_episode_mean_reward

        reward_diff = self.reward_difference
        if reward_diff is not None:
            reward_diff_tensor = torch.tensor(reward_diff)
            sensitivity = 10
            adjusted_dropout = torch.sigmoid(self.dropout_rate + sensitivity * reward_diff_tensor)
            self.dropout_rate = adjusted_dropout.item()
        else:
            reward_diff_tensor = torch.tensor(0)

        self.dropout_rate = max(0.1, min(self.dropout_rate, 0.9))
        
        return self.dropout_rate

dropout_manager = DynamicDropout()


class SimulatedAnnealingDropout:
    def __init__(self, initial_temp=0.5, 
                 decay_factor=0.9, 
                 min_temp=0.0001, 
                 update_interval=10,
                 convergence_threshold=0,##0.4
                 convergence_check_interval=1): ##977
        self.dropout_rate = 0.0  # Initialize dropout rate to zero
        self.temperature = initial_temp
        self.decay_factor = decay_factor
        self.min_temp = min_temp
        self.prev_mean_reward = float('-inf')
        self.update_interval = update_interval
        self.episode_count = 0
        self.convergence_count = 0
        self.convergence_threshold = convergence_threshold
        self.converged = False 
        self.convergence_check_interval = convergence_check_interval

        # Buffer to store last N episode rewards
        self.buffer = []
    
    def _acceptance_probability(self, old_cost, new_cost):
        reward_difference = new_cost - old_cost
        if reward_difference > 0:
            return 1.0
        return math.exp(reward_difference / self.temperature)
    
    def _check_convergence(self):
        # Calculate the standard deviation of the last N rewards
        std_dev = np.std(self.buffer)
        # Check if rewards have converged
        if std_dev > self.convergence_threshold:
            self.converged = True
        return self.converged

    def update_dropout_rate(self, episode_reward):
        # Increment episode count and convergence count
        self.episode_count += 1
        self.convergence_count += 1

        # Append the reward of the current episode to the buffer
        self.buffer.append(episode_reward)
        if len(self.buffer) > self.update_interval:
            self.buffer.pop(0)
        
        # Check for convergence if not already converged
        if not self.converged:
            if self.convergence_count >= self.convergence_check_interval:
                if self._check_convergence():
                    self.converged = True
                    self.convergence_count = 0  # Reset the convergence count for dropout updates
                    self.dropout_rate = 0.005  # Set dropout to 0.1 once convergence is detected
                else:
                    return self.dropout_rate
            else:
                return 0.0  # Return 0.0 until convergence_check_interval is reached

        # If convergence is detected, update dropout at intervals defined by self.update_interval
        if self.converged and self.episode_count >= self.update_interval:
            # Reset episode count for next interval
            self.episode_count = 0

            current_mean_reward = np.mean(self.buffer)

            # Perturb the dropout_rate to decrease slightly
            new_dropout_rate = self.dropout_rate - random.uniform(0.0002, 0.0002 * self.dropout_rate)
            new_dropout_rate = max(0.0005, min(new_dropout_rate, 0.1))

            # Calculate acceptance probability
            acceptance_prob = self._acceptance_probability(self.prev_mean_reward, current_mean_reward)

            # Decide if we should update dropout rate based on acceptance probability
            if acceptance_prob > random.uniform(0, 1):
                self.dropout_rate = new_dropout_rate
                self.prev_mean_reward = current_mean_reward

            # Decay the temperature
            self.temperature = max(self.min_temp, self.temperature * self.decay_factor)
            print(f"self.dropout_rate {self.dropout_rate}")
        
        return self.dropout_rate
