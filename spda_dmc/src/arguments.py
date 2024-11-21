import argparse
import numpy as np
import random


def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--eval_mode_1', default='color_hard', type=str)
	parser.add_argument('--only_encoder', default=True, type=str)
	parser.add_argument('--common_encoder', default=False, type=str)
	parser.add_argument('--double_aug', default=False, type=str)
	parser.add_argument('--aug_method', default='random_overlay', type=str)
	parser.add_argument('--if_obscat', default=True, type=str)
	parser.add_argument('--over_rand', default=False, type=str)


	parser.add_argument('--KL_encoder', default=False, type=str)
	parser.add_argument('--kl_lim', default=False, type=str)
	parser.add_argument('--kl_tar', default=0.25, type=int)
	parser.add_argument('--auto_para', default=False, type=str)
	parser.add_argument('--auto_Q', default=False, type=str)
	parser.add_argument('--encoder_para', default=0.02, type=float)
	parser.add_argument('--add_clip', default=False, type=str)
	parser.add_argument('--encoder_clip', default=0.2, type=float)
	parser.add_argument('--clip_step', default=500000, type=int)
	parser.add_argument('--critic_para', default=1, type=int)
	
	
	# agent
	parser.add_argument('--algorithm', default='spda', type=str)
	parser.add_argument('--train_steps', default='500000', type=str)    
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)
	parser.add_argument('--encoder_update_freq', default=1, type=int)
	parser.add_argument('--kl_beta', default=0.05, type=float)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# encoder-----------------------------------------------------
	parser.add_argument('--encoder_lr', default=0.0001, type=float)
	parser.add_argument('--encoder_beta', default=0.9, type=float)
	parser.add_argument('--encoder_multi', default=0, type=float)    # set 0
	# ------------------------------------------------------------

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--a_alpha', default=0.5, type=float)
	parser.add_argument('--b_beta', default=0.5, type=float)
	parser.add_argument('--g_gamma', default=0.5, type=float)

	# sgqn
	parser.add_argument('--sgqn_quantile', default=0.90, type=float)
	parser.add_argument("--svea_contrastive_coeff", default=0.1, type=float)
	parser.add_argument("--svea_norm_coeff", default=0.1, type=float)
	parser.add_argument("--attrib_coeff", default=0.25, type=float)
	parser.add_argument("--consistency", default=1, type=int)
	
	# madi
	parser.add_argument('--save_mask', default=False, action='store_true', help='save the masks used for MaDi')
	parser.add_argument('--masker_lr', default=1e-3, type=float)
	parser.add_argument('--masker_beta', default=0.9, type=float)
	parser.add_argument('--masker_num_layers', default=3, type=int)
	parser.add_argument('--masker_num_filters', default=32, type=int)
	parser.add_argument('--mask_type', default='soft', type=str, choices=['soft', 'hard', 'mixed'], help='either: `soft` (continuous), `hard` (binary), or `mixed`. If `hard`, then the mask is a binary mask. If `soft`, then the mask has continuous values between 0 and 1. If `mixed`, then the training env is soft masked and the test env is hard masked.')
	parser.add_argument('--mask_threshold', default=0.5, type=float, help='initial threshold (or quantile) for the mask. Only used if `mask_type` is not `soft`.')
	parser.add_argument('--mask_threshold_type', default='fix', type=str, choices=['fix', 'avg', 'quantile'], help='Only used if `mask_type` is not `soft`.')
	parser.add_argument('--augment', default='random_choose_double', type=str, choices=['none', 'conv', 'overlay', 'splice', 'random_choose_double'])
	parser.add_argument('--overlay_alpha', default=0.5, type=float, help='Opacity of the overlay augmentation')
	parser.add_argument('--save_aug', default=False, action='store_true', help='save the augmented observations during the critic update in MaDi')
	parser.add_argument('--save_aug_every', default=1e4, type=int, help='save augmented images every n steps')
	parser.add_argument('--save_all_frames', default=False, action='store_true', help='whether to save each frame of the observation or just the first one')
	
	# eval
	parser.add_argument('--save_freq', default='500k', type=str)
	parser.add_argument('--eval_freq', default='5000', type=str)
	parser.add_argument('--eval_episodes', default=5, type=int)
	parser.add_argument('--distracting_cs_intensity', default=0.5, type=float)

	# misc
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--de_num', default=0, type=int)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=False, action='store_true')
	parser.add_argument('--save_buffer', default=False, type=str)
	parser.add_argument('--buffer_save_freq', default=100000, type=int)

	args = parser.parse_args()

	assert args.algorithm in {'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea', 'madi', 'dtk', 'spda'}, f'specified algorithm "{args.algorithm}" is not supported'

	assert args.eval_mode_1 in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode_1}" is not supported'
	assert args.eval_mode_2 in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode_2}" is not supported'
	assert args.eval_mode_3 in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode_2}" is not supported'

	assert args.seed is not None, 'must provide seed for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))

	if args.eval_mode_1 == 'none':
		args.eval_mode_1 = None

	if args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
		args.image_size = 100
		args.image_crop_size = 84
	else:
		args.image_size = 84
		args.image_crop_size = 84
	
	return args
