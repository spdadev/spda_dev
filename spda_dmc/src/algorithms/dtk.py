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
import torchvision

class DTK(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.a_alpha = args.a_alpha
		self.b_beta = args.b_beta
		self.g_gamma = args.g_gamma
		self.double_aug = args.double_aug
		self.aug_method = args.aug_method
		self.tangent_loss = True
		self.de_num = args.de_num
		self.device = torch.device(f'cuda:{self.de_num}')


	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, obs_2=None, action_2=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs) 
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if not self.double_aug:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.a_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug_1 = augmentations.random_overlay(obs.clone())

			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action)
			critic_loss += self.b_beta * \
				(F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q))
			
			obs_aug_2 = augmentations.random_shift(obs.clone())
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_aug_2, action)
			critic_loss += self.g_gamma * \
				(F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.a_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug_1 = augmentations.random_conv_2(obs.clone())
			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_aug_1, action)
			critic_loss += self.b_beta * \
				(F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q))
			
			obs_aug_2 = augmentations.random_overlay(obs.clone())
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_aug_2, action)
			critic_loss += self.g_gamma * \
				(F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q))

			
		# if L is not None:
		# 	L.log('train_critic/loss', critic_loss, step)

		if self.tangent_loss:
			tangent_prop_loss = self.tangent_prop_loss(obs_aug_1, obs_aug_2, action, target_Q)
			critic_loss +=tangent_prop_loss

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		return obs_aug_1, obs_aug_2
			

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		obs_aug_1, obs_aug_2 = self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:

			obs_all = obs
			mu_ori, _, _, log_std_ori = self.actor(obs_all,compute_pi=False, compute_log_pi=False)

			mu_1, _, _, log_std_1 = self.actor(obs_aug_1,compute_pi=False, compute_log_pi=False)
			mu_2, _, _, log_std_2 = self.actor(obs_aug_2,compute_pi=False, compute_log_pi=False)

			kl_divergence_ori_aug_1 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_1, log_std_1)
			kl_divergence_ori_aug_2 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_2, log_std_2)

			kl_loss = (kl_divergence_ori_aug_1 + kl_divergence_ori_aug_2).mean()
			betta = 0.1

			self.update_actor_and_alpha(obs_all, L, step, kl_loss=kl_loss*betta)


		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

	def kl_divergence(self, mu_p, p_log_std, mu_q, q_log_std):
		p_std = torch.exp(p_log_std)
		q_std = torch.exp(q_log_std)


		kl = torch.log(q_std / p_std) + (p_std**2 + (mu_p - mu_q)**2) / (2 * q_std**2) - 0.5
		kl = torch.mean(kl, dim=-1)
		return kl
	
	def tangent_vector(self, obs):
		pad = nn.Sequential(torch.nn.ReplicationPad2d(1))
		pad_obs = pad(obs)
		index = np.random.randint(4, size=1)[0]
		if index == 0:
			# horizontal shift 1 pixel
			obs_aug = torchvision.transforms.functional.crop(pad_obs, top=1, left=2, height=obs.shape[-1], width=obs.shape[-1])
		elif index == 1:
			# horizontal shift 1 pixel
			obs_aug = torchvision.transforms.functional.crop(pad_obs, top=1, left=0, height=obs.shape[-1], width=obs.shape[-1])
		elif index == 2:
			# vertical shift 1 pixel
			obs_aug = torchvision.transforms.functional.crop(pad_obs, top=2, left=1, height=obs.shape[-1], width=obs.shape[-1])
		elif index == 3:
			# vertical shift 1 pixel
			obs_aug = torchvision.transforms.functional.crop(pad_obs, top=0, left=1, height=obs.shape[-1], width=obs.shape[-1])
		tan_vector = obs_aug - obs
		return tan_vector
	
	def tangent_prop_loss(self, obs_aug_1, obs_aug_2, action, target_Q):

		with torch.no_grad():
			# calculate the tangent vector
			tangent_vector1 = self.tangent_vector(obs_aug_1)
			tangent_vector2 = self.tangent_vector(obs_aug_2)

		obs_aug_1.requires_grad = True
		obs_aug_2.requires_grad = True
		# critic loss
		Q1_aug_1, Q2_aug_1 = self.critic(obs_aug_1, action)
		Q1_aug_2, Q2_aug_2 = self.critic(obs_aug_2, action)
		# critic_loss = F.mse_loss(Q1_aug_1, target_Q) + F.mse_loss(Q2_aug_1, target_Q)
		# critic_loss += F.mse_loss(Q1_aug_2, target_Q) + F.mse_loss(Q2_aug_2, target_Q)
		# # avg_critic_loss = critic_loss
		# avg_critic_loss = critic_loss / 2

		# add regularization for tangent prop
		# calculate the Jacobian matrix for non-linear model
		Q1 = torch.min(Q1_aug_1, Q2_aug_1)
		jacobian1 = torch.autograd.grad(outputs=Q1, inputs=obs_aug_1,
										grad_outputs=torch.ones(Q1.size(), device=self.device),
										retain_graph=True, create_graph=True)[0]
		Q2 = torch.min(Q1_aug_2, Q2_aug_2)
		jacobian2 = torch.autograd.grad(outputs=Q2, inputs=obs_aug_2,
										grad_outputs=torch.ones(Q2.size(), device=self.device),
										retain_graph=True, create_graph=True)[0]
		tan_loss1 = torch.mean(torch.square(torch.sum((jacobian1 * tangent_vector1), (3, 2, 1))), dim=-1)
		tan_loss2 = torch.mean(torch.square(torch.sum((jacobian2 * tangent_vector2), (3, 2, 1))), dim=-1)
		# tangent_prop_loss = tan_loss1*0.1
		tangent_prop_loss = (tan_loss1+tan_loss2)*0.1

		return tangent_prop_loss