from util import BaseCallback
from NerveNet.models.nerve_net_opt import DynamicDropout, SimulatedAnnealingDropout
import numpy as np

class RewardCallback(BaseCallback):

    def __init__(self, dropout_optimizer, manager):
        super(RewardCallback, self).__init__()
        
        self.rewards = []  # Initialize the rewards attribute
        self.dropout_optimizer = dropout_optimizer 
        self.prev_episode_mean_reward = None
        self.reward_difference = 0
        self.buffer = []
        self.manager = manager 
        self.log_interval = 5 # Set the logging interval
        self.episode_count = 0
        self.total_episodes = 3
        self.embedding_buffer =[]
    
    
    def _on_step(self):
        # if self.episode_count == self.total_episodes-1:  # Only for the last episode
        #     current_embedding = self.model.policy.mlp_extractor.save_all_embeddings()
        #     print(current_embedding)
        #     self.embedding_buffer.append(current_embedding)

        return True        
    
    def _on_rollout_end(self) -> None:
        if "rollout_buffer" in self.locals:
            episode_rewards = self.locals["rollout_buffer"].rewards
            self.rewards.extend(episode_rewards)

            # current_episode_mean_reward = np.mean(episode_rewards)
            # if self.prev_episode_mean_reward is not None:
            #     self.reward_difference = current_episode_mean_reward - self.prev_episode_mean_reward
            #     # Store this difference or use it further if needed

            # self.prev_episode_mean_reward = current_episode_mean_reward
            
            # if self.episode_count == self.total_episodes-1:
            #     print("start")
            #     with open('root_last_episode1.npy', 'wb') as f:
            #         print(self.embedding_buffer)
            #         np.save(f, np.array(self.embedding_buffer))
            #     print("end")
            # self.episode_count += 1  # Increment the episode count
            # self.embedding_buffer=[]

            
            # If episode count is a multiple of log_interval, log the SD
            if self.episode_count % self.log_interval == 0:
                reward_sd = np.std(self.rewards)  # Compute standard deviation
                self.logger.record("train/reward_sd", reward_sd)
                self.rewards = []  # Reset rewards to start collecting for next N episodes


        # new_dropout_rate = self.dropout_optimizer.update_dropout_rate(episode_rewards)
            new_dropout_rate = self.manager.update(episode_rewards)
            self.logger.record("train/dropout_rate", new_dropout_rate)
            self.model.policy.mlp_extractor.set_dropout_rate(new_dropout_rate)

        return episode_rewards