import glob
import pickle
import torch
from tqdm import tqdm
from torch.utils import data
import numpy as np


# Code for this dataloader is heavily borrowed from PECNet.
# https://github.com/HarshayuGirase/Human-Path-Prediction


'''for sanity check'''
def naive_social(p1_key, p2_key, all_data_dict):
	if abs(p1_key-p2_key)<4:
		return True
	else:
		return False

def find_min_time(t1, t2):
	'''given two time frame arrays, find then min dist (time)'''
	min_d = 9e4
	t1, t2 = t1[:8], t2[:8]

	for t in t2:
		if abs(t1[0]-t)<min_d:
			min_d = abs(t1[0]-t)

	for t in t1:
		if abs(t2[0]-t)<min_d:
			min_d = abs(t2[0]-t)

	return min_d

def find_min_dist(p1x, p1y, p2x, p2y):
	'''given two time frame arrays, find then min dist'''
	min_d = 9e4
	p1x, p1y = p1x[:8], p1y[:8]
	p2x, p2y = p2x[:8], p2y[:8]

	for i in range(len(p1x)):
		for j in range(len(p1x)):
			if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
				min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

	return min_d

def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
	p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
	p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
	p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
	p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]

	if find_min_time(p1_time, p2_time)>time_thresh:
		return False
	if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
		return False

	return True

def mark_similar(mask, sim_list):
	for i in range(len(sim_list)):
		for j in range(len(sim_list)):
			mask[sim_list[i]][sim_list[j]] = 1

def calc_interaction_nba(x):
	# x: (N,T,2)
	actor_num = x.shape[0]
	length = x.shape[1]
	# without ball interaction
	player = x[:-1,:,:]
	player_1 = player[None,:,:,:]
	player_2 = player[:,None,:,:]
	player_diff = player_2 - player_1
	player_dist = torch.norm(player_diff,dim=-1,p=2)
	player_mask = (player_dist<0.8)
	interaction_player = (torch.sum(player_mask)-(actor_num*length))/2

	ball = x[-1:,:,:]
	distence = torch.norm(player - ball,dim=-1,p=2)
	dist_mask = (distence<0.3)
	weight = 10
	interaction_ball = torch.sum(dist_mask) * weight

	# close_dist,close_player = torch.min(distence,dim=0)
	# close_player = (close_dist<0.3)*(close_player+1)
	# noempty_loc = (close_player != 0)
	# close_player = close_player[noempty_loc]
	# interaction_ball_player = torch.sum(((close_player[1:]-close_player[:-1])!=0))
	# weight = 20
	# interaction_ball_player = interaction_ball_player * weight
	return interaction_player + interaction_ball #+ interaction_ball_player


def collect_data(set_name, dataset_type = 'image', batch_size=512, time_thresh=48, dist_tresh=100, scene=None, verbose=True, root_path="./"):

	assert set_name in ['train','val','test']

	'''Please specify the parent directory of the dataset. In our case data was stored in:
		root_path/trajnet_image/train/scene_name.txt
		root_path/trajnet_image/test/scene_name.txt
	'''

	rel_path = '/trajnet_{0}/{1}/stanford'.format(dataset_type, set_name)

	full_dataset = []
	full_masks = []

	current_batch = []
	mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

	current_size = 0
	social_id = 0
	part_file = '/{}.txt'.format('*' if scene == None else scene)

	for file in glob.glob(root_path + rel_path + part_file):
		scene_name = file[len(root_path+rel_path)+1:-6] + file[-5]
		data = np.loadtxt(fname = file, delimiter = ' ')

		data_by_id = {}
		for frame_id, person_id, x, y in data:
			if person_id not in data_by_id.keys():
				data_by_id[person_id] = []
			data_by_id[person_id].append([person_id, frame_id, x, y])

		all_data_dict = data_by_id.copy()
		if verbose:
			print("Total People: ", len(list(data_by_id.keys())))
		while len(list(data_by_id.keys()))>0:
			related_list = []
			curr_keys = list(data_by_id.keys())

			if current_size<batch_size:
				pass
			else:
				full_dataset.append(current_batch.copy())
				mask_batch = np.array(mask_batch)
				full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])

				current_size = 0
				social_id = 0
				current_batch = []
				mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

			current_batch.append((all_data_dict[curr_keys[0]]))
			related_list.append(current_size)
			current_size+=1
			del data_by_id[curr_keys[0]]

			for i in range(1, len(curr_keys)):
				if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):
					current_batch.append((all_data_dict[curr_keys[i]]))
					related_list.append(current_size)
					current_size+=1
					del data_by_id[curr_keys[i]]

			mark_similar(mask_batch, related_list)
			social_id +=1


	full_dataset.append(current_batch)
	mask_batch = np.array(mask_batch)
	full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])
	return full_dataset, full_masks

def generate_pooled_data(b_size, t_tresh, d_tresh, train=True, scene=None, verbose=True):
	if train:
		full_train, full_masks_train = collect_data("train", batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		train = [full_train, full_masks_train]
		train_name = "../social_pool_data/train_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)
		with open(train_name, 'wb') as f:
			pickle.dump(train, f)

	if not train:
		full_test, full_masks_test = collect_data("test", batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		test = [full_test, full_masks_test]
		test_name = "../social_pool_data/test_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)# + str(b_size) + "_" + str(t_tresh) + "_" + str(d_tresh) + ".pickle"
		with open(test_name, 'wb') as f:
			pickle.dump(test, f)

def initial_pos(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,7,:].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches

def initial_pos_new(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,7,:].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches

def offset_pos(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,0,:].copy() #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches


def calculate_loss(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
	# reconstruction loss
	RCL_dest = criterion(x, reconstructed_x)
	# RCL_dest = criterion(x,interpolated_future)

	ADL_traj = criterion(future, interpolated_future) # better with l2 loss

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return RCL_dest, KLD, ADL_traj

def calculate_loss_mm(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
	# reconstruction loss
	batch = x.shape[0]
	actor_num = x.shape[1]
	interpolated_future = interpolated_future.contiguous().view(batch,actor_num,-1)
	reconstructed_x = reconstructed_x.contiguous().view(batch,actor_num,2)
	loss_ade = torch.mean(torch.sum((interpolated_future-future)**2,dim=2),dim=1)
	ADL_traj = loss_ade

	loss_fde = torch.mean(torch.sum((x-reconstructed_x)**2,dim=2),dim=1)
	RCL_dest = loss_fde

	# RCL_dest = criterion(x, reconstructed_x)
	# RCL_dest = criterion(x,interpolated_future)

	# ADL_traj = criterion(future, interpolated_future) # better with l2 loss

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return RCL_dest, KLD, ADL_traj

def calculate_loss_multi(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
	# reconstruction loss
	RCL_dest = criterion(x, reconstructed_x)
	# RCL_dest = criterion(x,interpolated_future)
	ADL_traj = criterion(future, interpolated_future)+5*criterion(future[:,:10], interpolated_future[:,:10])  \
				# + 10*criterion(future[:,:2], interpolated_future[:,:2]) \
				# + 8*criterion(future[:,:4], interpolated_future[:,:4]) \
				# + 6*criterion(future[:,:6], interpolated_future[:,:6]) \
				# + 4*criterion(future[:,:8], interpolated_future[:,:8]) \
			# + criterion(future[:,:10], interpolated_future[:,:10]) 
				# + criterion(future[:,:12], interpolated_future[:,:12]) \
				# + criterion(future[:,:14], interpolated_future[:,:14]) \
				# + criterion(future[:,:16], interpolated_future[:,:16])

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return RCL_dest, KLD, ADL_traj

def calculate_loss_double(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future, mean2,log_var2):
	# reconstruction loss
	RCL_dest = criterion(x, reconstructed_x)
	# RCL_dest = criterion(x,interpolated_future)

	ADL_traj = criterion(future, interpolated_future) # better with l2 loss

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	KLD_2 = -0.5 * torch.sum(1 + log_var2 - mean2.pow(2) - log_var2.exp())

	return RCL_dest, KLD, ADL_traj, KLD_2

class SocialDataset(data.Dataset):

	def __init__(self, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, scene=None, id=False, verbose=False):
		'Initialization'
		load_name = "./data/{0}_{1}{2}_{3}_{4}.pickle".format(set_name, 'all_' if scene is None else scene[:-2] + scene[-1] + '_', b_size, t_tresh, d_tresh)
		# print(load_name)
		with open(load_name, 'rb') as f:
			data = pickle.load(f)
		load_name_test = "./trajnet_image/test_trajnet.pkl"
		# with open(load_name_test, 'rb') as f:
		# 	data_test = pickle.load(f)
		# test_dataset = SceneDataset(data_test,
		# 							resize=1,#cfg['resize'],
		# 							total_len=20)
		# test_traj = test_dataset.trajectories
		# load_name_train = "./trajnet_image/train_trajnet.pkl"
		# with open(load_name_train, 'rb') as f:
		# 	data_train = pickle.load(f)
		traj, masks, scene_names = data
		assert len(traj) == 1
		# scene_names = []
		# for t in traj[0]:
		# 	row = data_test[(data_test['trackId'] == t[0][0]) & (data_test['frame'] == t[0][1])
		# 					& (data_test['x'] == t[0][2]) & (data_test['y'] == t[0][3])]
		# 	assert row.shape[0] == 1
		# 	scene_name = row['sceneId'].values[0]
		# 	scene_names.append(scene_name)
		# find row in data_test that has the same values as each item in traj
		# import ipdb; ipdb.set_trace()
		traj_new = []

		frame_ids = []
		ped_ids = []
		if id==False:
			for t in traj:
				t = np.array(t)
				frame_ids.append(t[:,:,1])
				ped_ids.append(t[:,:,0])
				t = t[:,:,2:]
				traj_new.append(t)
				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)
		else:
			for t in traj:
				t = np.array(t)
				traj_new.append(t)

				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)


		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name=="train":
				#add second time for the reversed tracklets...
				masks_new.append(m)
		
		seq_start_end_list = []
		for m in masks:
			total_num = m.shape[0]
			scene_start_idx = 0
			num_list = []
			for i in range(total_num):
				if i < scene_start_idx:
					continue
				scene_actor_num = np.sum(m[i])
				scene_start_idx += scene_actor_num 
				num_list.append(scene_actor_num)
			cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
			seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
			seq_start_end_list.append(seq_start_end)
			if set_name=="train":
				#add second time for the reversed tracklets...
				seq_start_end_list.append(seq_start_end)

		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)

		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches)) #for relative positioning
		self.seq_start_end_batches = seq_start_end_list
		self.scene_names = scene_names
		self.frame_ids = frame_ids
		self.ped_ids = ped_ids

		print("trajectory_batches.shape:", self.trajectory_batches.shape)
		print("mask_batches.shape:", self.mask_batches.shape)
		print("initial_pos_batches.shape:", self.initial_pos_batches.shape)
		print("len(seq_start_end_batches):", len(self.seq_start_end_batches[0]))

		if verbose:
			print("Initialized social dataloader...")


class SceneDataset(data.Dataset):
    def __init__(self, data, resize, total_len):
        """ Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""

        self.trajectories, self.meta, self.scene_list = self.split_trajectories_by_scene(
            data, total_len)
        self.trajectories = self.trajectories * resize

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        meta = self.meta[idx]
        scene = self.scene_list[idx]
        return trajectory, meta, scene

    def split_trajectories_by_scene(self, data, total_len):
        trajectories = []
        meta = []
        scene_list = []
        for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False),
                                     desc='Prepare Dataset'):
            trajectories.append(meta_df[['x', 'y'
                                        ]].to_numpy().astype('float32').reshape(
                                            -1, total_len, 2))
            meta.append(meta_df)
            scene_list.append(meta_df.iloc()[0:1].sceneId.item())
        return np.array(trajectories), meta, scene_list

class SocialDataset_new(data.Dataset):

	def __init__(self, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, scene=None, id=False, verbose=True):
		'Initialization'
		load_name = "./data/{0}_{1}{2}_{3}_{4}.pickle".format(set_name, 'all_' if scene is None else scene[:-2] + scene[-1] + '_', b_size, t_tresh, d_tresh)
		print(load_name)
		with open(load_name, 'rb') as f:
			data = pickle.load(f)

		traj, masks, scene_names = data
		traj_new = []

		scene_names_new = []
		for s in scene_names:
			scene_names_new.append(s)

		if id==False:
			for t in traj:
				t = np.array(t)
				t = t[:,:,2:]
				traj_new.append(t)
				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)
		else:
			for t in traj:
				t = np.array(t)
				traj_new.append(t)

				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)


		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name=="train":
				#add second time for the reversed tracklets...
				masks_new.append(m)
		
		seq_start_end_list = []
		for m in masks:
			total_num = m.shape[0]
			scene_start_idx = 0
			num_list = []
			for i in range(total_num):
				if i < scene_start_idx:
					continue
				scene_actor_num = np.sum(m[i])
				scene_start_idx += scene_actor_num 
				num_list.append(scene_actor_num)
			cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
			seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
			seq_start_end_list.append(seq_start_end)
			if set_name=="train":
				#add second time for the reversed tracklets...
				seq_start_end_list.append(seq_start_end)

		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)

		scene_names_new = np.array(scene_names_new)
		self.scene_names = scene_names_new.copy()
		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.initial_pos_batches = np.array(initial_pos_new(self.trajectory_batches)) #for relative positioning
		self.seq_start_end_batches = seq_start_end_list
		self.offset_batches = np.array(offset_pos(self.trajectory_batches))
		if verbose:
			print("Initialized social dataloader...")
