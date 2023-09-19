import os
import datetime
import torch
import torch.nn as nn
import numpy as np
from models.model_test_trajectory_res import model_encdec

from sddloader import *

torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):

        self.save_trajectories_path = config.save_trajectories_path

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'testing/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
 
       
        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

       
        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.settings = {
            "train_batch_size": config.train_b_size,
            "test_batch_size": config.test_b_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 12,
        }

        # model
        self.model_ae = torch.load(config.model_ae, map_location=torch.device('cpu')).cuda()
        self.mem_n2n = model_encdec(self.settings, self.model_ae)
        

        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        


    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0
    
    
    def fit(self):
        
        dict_metrics_test = self.evaluate(self.test_dataset)
        print('Test FDE_48s: {} ------ Test ADE: {}'.format(dict_metrics_test['fde_48s'], dict_metrics_test['ade_48s']))
        print('-'*100)

    @staticmethod
    def flatten_scene(trajs_arr, frame_nums=None, ped_nums=None, frame_skip=10):
        """flattens a 3d scene of shape (num_peds, ts, 2) into a 2d array of shape (num_peds, ts x 4)
        ped_nums (optional): list of ped numbers to assign, length == num_peds
        frame_nums (optional): list of frame numbers to assign to the resulting, length == ts
        """
        if ped_nums is None:
            ped_nums = np.arange(0, trajs_arr.shape[1])
        if frame_nums is None:
            frame_nums = np.arange(0, trajs_arr.shape[0] * frame_skip, frame_skip)
        ped_ids = np.tile(np.array(ped_nums).reshape(-1, 1), (1, trajs_arr.shape[1])).reshape(-1, 1)
        frame_ids = np.tile(np.array(frame_nums).reshape(1, -1), (trajs_arr.shape[0], 1)).reshape(-1, 1)
        trajs_arr = np.concatenate([frame_ids, ped_ids, trajs_arr.reshape(-1, 2)], -1)
        return trajs_arr

    @staticmethod
    def save_trajectories(trajectory, save_dir, seq_name, frame, suffix=''):
        """Save trajectories in a text file.
        Input:
            trajectory: (np.array/torch.Tensor) Predcited trajectories with shape
                        of (future_timesteps * n_pedestrian, 4). The last element is
                        [frame_id, track_id, x, y] where each element is float.
            save_dir: (str) Directory to save into.
            seq_name: (str) Sequence name (e.g., eth_biwi, coupa_0)
            frame: (num) Frame ID.
            suffix: (str) Additional suffix to put into file name.
        """

        fname = f"{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt"
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        np.savetxt(fname, trajectory, fmt="%s")

    def evaluate(self, dataset):
        
        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}

        with torch.no_grad():
            for i, (traj, mask, initial_pos, seq_start_end, all_seq_names, all_frame_ids, all_ped_ids) \
                in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches,
                                 dataset.seq_start_end_batches, [dataset.scene_names], dataset.frame_ids, dataset.ped_ids)):
                traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), \
                    torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)
                # traj (B, T, 2)
                print(f'doing batch {i}')
                initial_pose = traj[:, 7, :] / 1000
                
                traj_norm = traj - traj[:, 7:8, :]
                x = traj_norm[:, :self.config.past_len, :]
                destination = traj_norm[:, -2:, :]
                

                abs_past = traj[:, :self.config.past_len, :]
                output = self.mem_n2n(x, abs_past, seq_start_end, initial_pose)
                output = output.data
                # B, K, t, 2

                seq_name_to_frames = {}
                seq_names = [ 'coupa_0', 'coupa_1', 'gates_2', 'hyang_0', 'hyang_1', 'hyang_3',
                              'hyang_8', 'little_0', 'little_1', 'little_2', 'little_3',
                              'nexus_5', 'nexus_6', 'quad_0', 'quad_1', 'quad_2', 'quad_3', ]
                last_frame = traj[:, 7:8, :]
                output = output + last_frame.unsqueeze(1)
                output = output.cpu().numpy()
                labels = traj[:, self.config.past_len:, :].cpu().numpy()
                obses = traj[:, :self.config.past_len, :].cpu().numpy()
                for seq in seq_names:
                    idxs = np.where(np.array(dataset.scene_names) == seq)[0]
                    out = output[idxs]
                    lbl = labels[idxs]
                    obs = obses[idxs]
                    fram = all_frame_ids[idxs]
                    ped = all_ped_ids[idxs]
                    # gather same frames together
                    unique_final_obs_frames_sorted = sorted(np.unique(fram[:, 7]))
                    frames_this_seq = []
                    for frame in unique_final_obs_frames_sorted:
                        idxs = np.where(fram[:, 7] == frame)[0]  # 7 is the last obs step
                        assert len(idxs) > 0
                        outp = out[idxs]
                        labl = lbl[idxs]
                        obse = obs[idxs]
                        fra = fram[idxs]
                        pede = ped[idxs]
                        frames_this_seq.append([outp, labl, obse, fra, pede])
                    seq_name_to_frames[seq] = frames_this_seq

                # save trajectories
                save_path = self.save_trajectories_path
                if save_path != "":
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print("saving trajs to:", save_path)

                    for seq_name, frames in seq_name_to_frames.items():
                        for out, lbl, obs, frame_ids, ped_ids in tqdm(frames, desc=f'Saving trajs in {seq_name}...', total=len(frames)):
                            assert np.all([np.all(frame_ids[0] == f for f in frame_ids)]), \
                                f'all 20 frame_ids should be the same for all peds but are {frame_ids}'
                            frame_ids = frame_ids[0]
                            ped_ids = ped_ids[:,0]

                            # obs --> (num_peds, 8, 2)
                            dset_outputs = out
                            dset_labels = lbl  # --> (num_peds, 12, 2)

                            # check that the seqs metadata and agents match
                            pred = dset_outputs.transpose(1, 0, 2, 3)  # --> (20, num_peds, 12, 2)
                            flattened_gt = self.flatten_scene(dset_labels, frame_ids[self.config.past_len:], ped_ids)
                            flattened_obs = self.flatten_scene(obs, frame_ids[:self.config.past_len], ped_ids)

                            # todo save correct ped_ids and frame_ids
                            for sample_i, sample in enumerate(pred):
                                flattened_peds = self.flatten_scene(sample, frame_ids[self.config.past_len:], ped_ids)
                                self.save_trajectories(flattened_peds, save_path, seq_name, frame_ids[self.config.past_len - 1],
                                                       suffix=f'/sample_{sample_i:03d}')
                            self.save_trajectories(flattened_gt, save_path, seq_name, frame_ids[self.config.past_len - 1],
                                                   suffix='/gt')
                            self.save_trajectories(flattened_obs, save_path, seq_name, frame_ids[self.config.past_len - 1],
                                                   suffix='/obs')

                print(f"Done saving trajectories to {save_path}")
                exit()

                future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(output - future_rep, dim=3)
                mean_distances = torch.mean(distances[:, :, -1:], dim=2)
                index_min = torch.argmin(mean_distances, dim=1)
                min_distances = distances[torch.arange(0, len(index_min)), index_min]

                fde_48s += torch.sum(min_distances[:, -1])
                ade_48s += torch.sum(torch.mean(min_distances, dim=1))
                samples += distances.shape[0]


            dict_metrics['fde_48s'] = fde_48s / samples
            dict_metrics['ade_48s'] = ade_48s / samples

        return dict_metrics
