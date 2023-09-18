import os
import math
import datetime
from random import sample
import numpy as np

import torch
import torch.nn as nn

from models.model_test_trajectory import MemoNet

from data.dataloader import data_generator
from utils.config import Config
from utils.utils import prepare_seed, print_log

torch.set_num_threads(5)


class Trainer:
	def __init__(self, config):
		"""
		The Trainer class handles the training procedure for training the autoencoder.
		:param config: configuration parameters (see train_ae.py)
		"""
		
		self.cfg = Config(config.cfg, config.info, config.tmp, create_dirs=True)
		self.save_trajectories_path = config.save_trajectories_path
		torch.set_default_dtype(torch.float32)
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')
		self.train_generator = data_generator(self.cfg, self.log, split='train', phase='training')
		self.eval_generator = data_generator(self.cfg, self.log, split='val', phase='testing')
		self.test_generator = data_generator(self.cfg, self.log, split='test', phase='testing')

		self.max_epochs = self.cfg.num_epochs
		# model
		self.MemoNet = MemoNet(self.cfg)
		self.MemoNet.model_encdec.load_state_dict(torch.load(self.cfg.model_encdec, map_location='cpu'))
		# loss
		self.criterionLoss = nn.MSELoss()

		self.opt = torch.optim.Adam(self.MemoNet.parameters(), lr=self.cfg.lr)
		self.iterations = 0
		if self.cfg.cuda:
			self.criterionLoss = self.criterionLoss.cuda()
			self.MemoNet = self.MemoNet.cuda()
		

	def fit(self):
		dict_metrics_test = self.evaluate(self.test_generator)
		print_log('------ Test FDE_48s: {} ------ Test ADE: {}'.format(dict_metrics_test['fde_48s'], dict_metrics_test['ade_48s']), log=self.log)
			 
	def rotate_traj(self, past, future, past_abs):
		past_diff = past[:, 0]
		past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
		past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

		rotate_matrix = torch.zeros((past_theta.size(0), 2, 2)).to(past_theta.device)
		rotate_matrix[:, 0, 0] = torch.cos(past_theta)
		rotate_matrix[:, 0, 1] = torch.sin(past_theta)
		rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
		rotate_matrix[:, 1, 1] = torch.cos(past_theta)

		past_after = torch.matmul(rotate_matrix, past.transpose(1, 2)).transpose(1, 2)
		future_after = torch.matmul(rotate_matrix, future.transpose(1, 2)).transpose(1, 2)
		
		b1 = past_abs.size(0)
		b2 = past_abs.size(1)
		for i in range(b1):
			past_diff = (past_abs[i, 0, 0]-past_abs[i, 0, -1]).unsqueeze(0).repeat(b2, 1)
			past_theta = torch.atan(torch.div(past_diff[:, 1], past_diff[:, 0]+1e-5))
			past_theta = torch.where((past_diff[:, 0]<0), past_theta+math.pi, past_theta)

			rotate_matrix = torch.zeros((b2, 2, 2)).to(past_theta.device)
			rotate_matrix[:, 0, 0] = torch.cos(past_theta)
			rotate_matrix[:, 0, 1] = torch.sin(past_theta)
			rotate_matrix[:, 1, 0] = - torch.sin(past_theta)
			rotate_matrix[:, 1, 1] = torch.cos(past_theta)
			# print(past_abs.size())
			past_abs[i] = torch.matmul(rotate_matrix, past_abs[i].transpose(1, 2)).transpose(1, 2)
		# print('-'*50)
		# print(past_abs.size())
		return past_after, future_after, past_abs

	@staticmethod
	def save_trajectories(trajectory, save_dir, seq_name, frame, suffix=''):
		"""Save trajectories in a text file.
        Input:
            trajectory: (np.array/torch.Tensor) Predcited trajectories with shape
                        of (n_pedestrian, future_timesteps, 4). The last elemen is
                        [frame_id, track_id, x, y] where each element is float.
            save_dir: (str) Directory to save into.
            seq_name: (str) Sequence name (e.g., eth_biwi, coupa_0)
            frame: (num) Frame ID.
            suffix: (str) Additional suffix to put into file name.
        """
		fname = f"{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt"
		if not os.path.exists(os.path.dirname(fname)):
			os.makedirs(os.path.dirname(fname))

		if isinstance(trajectory, torch.Tensor):
			trajectory = trajectory.cpu().numpy()
		np.savetxt(fname, trajectory, fmt="%.3f")

	@staticmethod
	def format_agentformer_trajectories(trajectory, data, cfg, timesteps=12, frame_scale=10, future=True):
		formatted_trajectories = []
		if not future:
			trajectory = torch.flip(trajectory, [0, 1])
		for i, track_id in enumerate(data['valid_id']):
			if data['pred_mask'] is not None and data['pred_mask'][i] != 1.0:
				continue
			for j in range(timesteps):
				if future:
					curr_data = data['fut_data'][j]
				else:
					curr_data = data['pre_data'][j]
				# Get data with the same track_id
				updated_data = curr_data[curr_data[:, 1] == track_id].squeeze()
				if cfg.dataset in [
						'eth', 'hotel', 'univ', 'zara1', 'zara2', 'gen',
						'real_gen', 'adversarial'
				]:
					# [13, 15] correspoinds to the 2D position
					updated_data[[13, 15]] = trajectory[i, j].cpu().numpy()
				elif 'sdd' in cfg.dataset:
					updated_data[[2, 3]] = trajectory[i, j].cpu().numpy()
				else:
					raise NotImplementedError()
				formatted_trajectories.append(updated_data)
		if len(formatted_trajectories) == 0:
			return np.array([])

		# Convert to numpy array and get [frame_id, track_id, x, y]
		formatted_trajectories = np.vstack(formatted_trajectories)
		if cfg.dataset in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
			formatted_trajectories = formatted_trajectories[:, [0, 1, 13, 15]]
			formatted_trajectories[:, 0] *= frame_scale
		elif cfg.dataset == 'trajnet_sdd':
			formatted_trajectories[:, 0] *= frame_scale

		if not future:
			formatted_trajectories = np.flip(formatted_trajectories, axis=0)

		return formatted_trajectories

	def evaluate(self, generator):
		prepare_seed(self.cfg.seed)
		ade_48s = fde_48s = 0
		samples = 0
		dict_metrics = {}
		with torch.no_grad():
			count = 0
			while not generator.is_epoch_end():
				
				data = generator()
				if data is not None:
					past = torch.stack(data['pre_motion_3D']).cuda()
					future = torch.stack(data['fut_motion_3D']).cuda()
					last_frame = past[:, -1:]
					past_normalized = past - last_frame
					fut_normalized = future - last_frame
					

					past_abs = past.unsqueeze(0).repeat(past.size(0), 1, 1, 1)
					past_centroid = past[:, -1:, :].unsqueeze(1)
					past_abs = past_abs - past_centroid

					scale = 1
					if self.cfg.scale.use:
						scale = torch.mean(torch.norm(past_normalized[:, 0], dim=1)) / 3
						if scale<self.cfg.scale.threshold:
							scale = 1
						else:
							if self.cfg.scale.type == 'divide':
								scale = scale / self.cfg.scale.large
							elif self.cfg.scale.type == 'minus':
								scale = scale - self.cfg.scale.large
						if self.cfg.scale.type=='constant':
							scale = self.cfg.scale.value
						past_normalized = past_normalized / scale
						past_abs = past_abs / scale

					if self.cfg.rotation:
						past_normalized, fut_normalized, past_abs = self.rotate_traj(past_normalized, fut_normalized, past_abs)
					end_pose = past_abs[:, :, -1]

					prediction = self.MemoNet(past_normalized, past_abs, end_pose)
					prediction = prediction.data * scale

					save_dir = self.save_trajectories_path
					if save_dir != "":
						data['frame_scale'] = 10
						frame = data['frame'] * data['frame_scale']
						print(f"dataset: {self.cfg.dataset} frame:", frame)
						gt_motion = fut_normalized + last_frame
						obs_motion = past_normalized + last_frame
						pred_motion = prediction + last_frame.unsqueeze(1)
						for idx, sample in enumerate(pred_motion.transpose(0, 1)):
							formatted = self.format_agentformer_trajectories(sample, data, self.cfg, timesteps=12,
																		frame_scale=data['frame_scale'], future=True)
							self.save_trajectories(formatted, save_dir, data['seq'], frame, suffix=f"/sample_{idx:03d}")
						formatted = self.format_agentformer_trajectories(gt_motion, data, self.cfg, timesteps=12,
																		 frame_scale=data['frame_scale'], future=True)
						self.save_trajectories(formatted, save_dir, data['seq'], frame, suffix='/gt')
						formatted = self.format_agentformer_trajectories(obs_motion, data, self.cfg, timesteps=8,
																		 frame_scale=data['frame_scale'], future=False)
						self.save_trajectories(formatted, save_dir, data['seq'], frame, suffix="/obs")

					future_rep = fut_normalized.unsqueeze(1).repeat(1, 20, 1, 1)
					distances = torch.norm(prediction - future_rep, dim=3)
					distances = torch.where(torch.isnan(distances), torch.full_like(distances, 10), distances)
					# N, K, T

					mean_distances = torch.mean(distances[:, :, -1:], dim=2)
					mean_distances_ade = torch.mean(distances, dim=2)

					index_min = torch.argmin(mean_distances, dim=1)
					min_distances = distances[torch.arange(0, len(index_min)), index_min]

					index_min_ade = torch.argmin(mean_distances_ade, dim=1)
					min_distances_ade = distances[torch.arange(0, len(index_min_ade)), index_min_ade]

					fde_48s += torch.sum(min_distances[:, -1])
					ade_48s += torch.sum(torch.mean(min_distances_ade, dim=1))

					samples += distances.shape[0]
		dict_metrics['fde_48s'] = fde_48s / samples
		dict_metrics['ade_48s'] = ade_48s / samples

		return dict_metrics
