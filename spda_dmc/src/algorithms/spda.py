import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
import numpy as np
import time

class SPDA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.double_aug = args.double_aug
		self.a_alpha = args.a_alpha
		self.b_beta = args.b_beta
		self.g_gamma = args.g_gamma
		if self.double_aug:
			self.b_beta /= 2
			self.g_gamma /= 2
		self.aug_method = args.aug_method
		self.over_rand = args.over_rand
		if self.over_rand:
			self.overlay = augmentations.random_overlay_rand
		else:
			self.overlay = augmentations.random_overlay
		self.encoder_update_freq = args.encoder_update_freq
		self.only_encoder = args.only_encoder

		self.eval_freq = args.eval_freq
		self.kl_beta = args.kl_beta
		self.kl_lim = args.kl_lim
		self.kl_tar = args.kl_tar
		self.encoder_para = args.encoder_para
		self.add_clip = args.add_clip
		self.encoder_clip = args.encoder_clip
		self.clip_step = args.clip_step
		self.critic_para = args.critic_para
		self.auto_para = args.auto_para
		self.auto_Q = args.auto_Q
		self.add_VRE = args.add_VRE
		self.if_obscat = args.if_obscat

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, obs_2=None, action_2=None, update_encoder=False):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)    
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if not self.double_aug:

			obs_1_aug_1 = augmentations.random_overlay(obs.clone())
			obs = utils.cat(obs, obs_aug_1)
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)    
			critic_loss = (self.a_alpha + self.b_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))    
			
 
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.a_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_1_aug_1 = augmentations.random_conv_2(obs.clone())

			current_Q1_aug_1, current_Q2_aug_1 = self.critic(obs_1_aug_1, action)
			critic_loss += self.b_beta * \
				(F.mse_loss(current_Q1_aug_1, target_Q) + F.mse_loss(current_Q2_aug_1, target_Q))
			
			obs_1_aug_2 = augmentations.random_overlay(obs.clone())
			current_Q1_aug_2, current_Q2_aug_2 = self.critic(obs_1_aug_2, action)
			critic_loss += self.g_gamma * \
				(F.mse_loss(current_Q1_aug_2, target_Q) + F.mse_loss(current_Q2_aug_2, target_Q))	
				
		critic_loss = critic_loss*self.critic_para


		obs_aug_1 = obs_1_aug_1
		obs_aug_2 = obs_1_aug_2

		if update_encoder:
			if not self.double_aug:
				loss_encoder = self.update_encoder(obs_2, obs_1_aug_1)
			else:
				loss_encoder = self.update_encoder(obs_2, obs_1_aug_1, obs_1_aug_2)

		if self.add_clip and step <= self.clip_step:
			with torch.no_grad():
				lim = critic_loss*self.encoder_clip
				ratio = 1
				if abs(loss_encoder) > lim:
					ratio = lim / abs(loss_encoder)
			loss_encoder = loss_encoder * ratio
		
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		(critic_loss+loss_encoder).backward()
		self.critic_optimizer.step()
	
		# --------------------------------------------------------------------


		return obs_aug_1, obs_aug_2, critic_loss, loss_encoder
		# --------------------------------------------------------------------

	def update_encoder(self, obs_2, obs_1_aug_1 = None, obs_1_aug_2 = None):
		if not self.double_aug:
			if obs_2 is not None and len(obs_2) != 0:
				obs_2_aug_1 = self.overlay(obs_2.clone())
				obs_aug_1 = torch.cat((obs_1_aug_1, obs_2_aug_1), dim=0)
			else:
				obs_aug_1 = obs_1_aug_1
				
			emb_aug_1 = self.critic.encoder(obs_aug_1)
			cov_matrix = torch.cov(emb_aug_1.T)
			loss_encoder = -torch.trace(cov_matrix)

		else:
			if obs_2 is not None and len(obs_2) != 0:
				obs_2_aug_1 = self.overlay(obs_2.clone())
				obs_2_aug_2 = augmentations.random_conv(obs_2.clone())
				obs_aug_1 = torch.cat((obs_1_aug_1, obs_2_aug_1), dim=0)
				obs_aug_2 = torch.cat((obs_1_aug_2, obs_2_aug_2), dim=0)
			else:
				obs_aug_1 = obs_1_aug_1
				obs_aug_2 = obs_1_aug_2

			emb_aug_1 = self.critic.encoder(obs_aug_1)
			emb_aug_2 = self.critic.encoder(obs_aug_2)
			emb_aug_all = torch.cat((emb_aug_1, emb_aug_2), dim=0)
			cov_matrix = torch.cov(emb_aug_all.T)
			loss_encoder = -torch.trace(cov_matrix)
		
		return loss_encoder*self.encoder_para


	def update(self, replay_buffer, L, step, cof = 1):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()
		obs_2, action_2 = replay_buffer.sample_encoder()
		update_encoder = True
		kl_loss = None
		
		obs_aug_1, obs_aug_2, critic_loss, loss_encoder = self.update_critic(obs, action, reward, next_obs, not_done, L, step, obs_2, action_2, update_encoder)

		if not self.only_encoder and (step+1) % self.eval_freq == 0:

			mu_ori, _, _, log_std_ori = self.actor(obs,compute_pi=False, compute_log_pi=False)
			mu_1, _, _, log_std_1 = self.actor(obs_aug_1[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
			kl_divergence_ori_aug_1 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_1, log_std_1)
			kl_loss = (kl_divergence_ori_aug_1).mean()

			if obs_aug_2 is not None:
				mu_2, _, _, log_std_2 = self.actor(obs_aug_2[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
				kl_divergence_ori_aug_2 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_2, log_std_2)
				kl_loss = kl_loss + (kl_divergence_ori_aug_2).mean()

		if step % self.actor_update_freq == 0:
			if self.if_obscat:
				if obs_aug_2 is not None:
					obs_act = torch.cat((obs, obs_aug_1[0:obs.size(0)], obs_aug_2[0:obs.size(0)]), dim=0)
				else:
					obs_act = torch.cat((obs, obs_aug_1[0:obs.size(0)]), dim=0)
			else:
				obs_act = obs
			if self.only_encoder:    # True
				self.update_actor_and_alpha(obs_act, L, step)
			else:    # add KL_loss to actor
				mu_ori, _, _, log_std_ori = self.actor(obs,compute_pi=False, compute_log_pi=False)
				mu_1, _, _, log_std_1 = self.actor(obs_aug_1[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
				kl_divergence_ori_aug_1 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_1, log_std_1)
				kl_loss = (kl_divergence_ori_aug_1).mean()

				if obs_aug_2 is not None:
					mu_2, _, _, log_std_2 = self.actor(obs_aug_2[0:obs.size(0)],compute_pi=False, compute_log_pi=False)
					kl_divergence_ori_aug_2 = self.kl_divergence(mu_ori.detach(), log_std_ori.detach(), mu_2, log_std_2)
					kl_loss = kl_loss + (kl_divergence_ori_aug_2).mean()

				betta = self.kl_beta
				if self.kl_lim:
					if kl_loss < self.kl_tar:
						betta = 0
					
				self.update_actor_and_alpha(obs_act, L, step, kl_loss=kl_loss*betta)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		return kl_loss, critic_loss, loss_encoder

	def kl_divergence(self, mu_p, p_log_std, mu_q, q_log_std):
		p_std = torch.exp(p_log_std)
		q_std = torch.exp(q_log_std)

		kl = torch.log(q_std / p_std) + (p_std**2 + (mu_p - mu_q)**2) / (2 * q_std**2) - 0.5
		kl = torch.mean(kl, dim=-1)
		return kl
