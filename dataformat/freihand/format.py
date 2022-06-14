import skimage.io as io
import numpy as np
import time
import os
import json
from pathlib import Path
import shutil

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560*4  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'

def load_db_annotation(root, dataType):
    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(root, '%s_K.json' % dataType)
    mano_path = os.path.join(root, '%s_mano.json' % dataType)
    xyz_path = os.path.join(root, '%s_xyz.json' % dataType)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return zip(K_list, mano_list, xyz_list)

def load_image(root, dataType):
    path = os.path.join(root, dataType, 'rgb',
                            '%08d.jpg' % id)
    return io.imread(path)

def load_keypoints(ann):
    K, _, xyz = ann
    return projectPoints(xyz, K)

def save(paths, root, name, ext):
    savepath = root + "0/"
    Path(savepath).mkdir(parents=True, exist_ok=True)
    for path in paths:
        if os.path.exists(path):
            shutil.copy(path, savepath + name + str(len(os.listdir(savepath))) + ext)

def save_keypoints(anns, root, name):
    savepath = root + "0/"
    Path(savepath).mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        for ann in anns:
            keypoints = load_keypoints(ann)
            keypoints = np.c_[keypoints, np.ones(keypoints.shape[0])]
            pose = {
                'people': [
                    {
                        'hand_right_keypoints_2d': keypoints.flatten().tolist() ,
                        'hand_left_keypoints_2d': [] 
                    }
                ]
            }
            with open(savepath + name + str(len(os.listdir(savepath))) + '.json', 'w') as f:
                json.dump(pose, f)

def get_paths(root, type):
    return [os.path.join(root, type, 'rgb', '%08d.jpg' % id) for id in range(db_size(type))]

root = './'
train_anns = [(np.array(K), np.array(mano), np.array(xyz)) for K, mano, xyz in load_db_annotation(root, 'training')]
valid_anns = [(np.array(K), np.array(mano), np.array(xyz)) for K, mano, xyz in load_db_annotation(root, 'evaluation')]
train_paths = get_paths(root, 'training')
valid_paths = get_paths(root, 'evaluation')
# save(train_paths, './train/', 'train', '.jpg')
# save(valid_paths, './valid/', 'valid', '.jpg')
save_keypoints(train_anns, './train_poses/', 'train')
save_keypoints(valid_anns, './valid_poses/', 'valid')