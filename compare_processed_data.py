"compare processed data from pecnet, y-net, and memonet"
import os
import numpy as np
import pickle as pkl

def get_seq_start_end_list(masks):
    for m_row in masks:
        num_peds = m_row.shape[0]
        scene_start_idx = 0
        num_peds_in_scene = []
        for i in range(num_peds):
            if i < scene_start_idx:
                continue
            scene_actor_num = np.sum(m_row[i])
            scene_start_idx += scene_actor_num
            num_peds_in_scene.append(scene_actor_num)
            assert scene_actor_num + i == scene_start_idx
        cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
        seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        seq_start_end_list.append(seq_start_end)
    return seq_start_end_list

def extract_blocks_(masks):
    seq_start_end_list = []
    blocks = []
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
            assert scene_actor_num + i == scene_start_idx
            blocks.append(m[i:i + scene_actor_num, i:i + scene_actor_num])
        cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
        seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        seq_start_end_list.append(seq_start_end)
    return seq_start_end_list, blocks

def main():
    # test data
    with open('data/pecnet_data_sdd/test_all_4096_0_100.pickle', 'rb') as f:
        pecnet_test = pkl.load(f)
        # tuple of 1 list of lists of lists of lists: (1, 2829, 20, 4), (1, 2829, 2829)
        # (bs, num_peds, num_frames, 4), mask matrix
    with open('data/ynet_sdd/train_trajnet.pkl', 'rb') as f:
        ynet_test = pkl.load(f)
        # (169880, 6) dataframe
        # frame, trackId, x, y, sceneId, metaId
        print("num_peds:", len(ynet_test['trackId'].unique()))
        # 1050
        print(f"ynet_test.shape: {ynet_test.shape}")
    with open('data/test_all_4096_0_100.pickle', 'rb') as f:
        memonet_test = pkl.load(f)
        print(f"len(memonet_test): {len(memonet_test)}")
        # same exact as PECNet
        # tuple of 1 list of lists of lists of lists: (1, 2829, 20, 4), (1, 2829, 2829)
    traj, mask = pecnet_test
    seq_se, extracted_blocks = list(extract_blocks_(mask))
    print(f"len(extracted_blocks): {len(extracted_blocks)}")
    print(f"extracted_blocks: {extracted_blocks[0:100]}")

    import ipdb;
    ipdb.set_trace()
    with open('data/pecnet_data_sdd/train_all_512_0_100.pickle', 'rb') as f:
        pecnet_train = pkl.load(f)
    with open('data/ynet_sdd/train_trajnet.pkl', 'rb') as f:
        ynet_train = pkl.load(f)
    with open('data/train_all_512_0_100.pickle', 'rb') as f:
        memonet_train = pkl.load(f)
        print("memonet_train.shape:", memonet_train.shape)
        print("len(memonet_train):", len(memonet_train))


if __name__ == "__main__":
    main()