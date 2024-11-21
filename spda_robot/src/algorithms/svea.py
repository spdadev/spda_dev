import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class SVEA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.a_alpha = args.a_alpha
		self.b_beta = args.b_beta
		self.g_gamma = args.g_gamma


	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, obs_2=None, action_2=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs) 
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.a_alpha == self.b_beta:
			obs = utils.cat(obs, augmentations.random_conv(obs.clone()))  
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)    
			critic_loss = (self.a_alpha + self.b_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))    
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.a_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug_1 = augmentations.random_conv(obs.clone())
			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action)
			critic_loss += self.b_beta * \
				(F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q))
			
			obs_aug_2 = augmentations.random_overlay(obs.clone())
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_aug_2, action)
			critic_loss += self.g_gamma * \
				(F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q))

			
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
			

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()    # obs.shape:128*9*84*84

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:

			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

