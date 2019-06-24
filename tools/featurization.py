import _init_paths
import tensorflow as tf
from fcn.config import cfg, cfg_from_file
from fcn.test_features import getYCBExtents, getYCBPoints, feature_net_single_frame, ycb_num_classes
from fcn.test import Voxelizer

from object_pose_utils.utils import to_np, to_var
import cv2
import numpy as np
import scipy.io as scio
from quat_math import quaternion_from_matrix

def getObjectGTQuaternion(meta_data, obj):
    meta_idx = np.where(meta_data['cls_indexes'].flatten()==obj)[0]
    if(len(meta_idx) == 0):
        return None
    else:
        meta_idx = meta_idx[0]
    target_r = meta_data['poses'][:, :, meta_idx][:, 0:3]
    target_t = np.array([meta_data['poses'][:, :, meta_idx][:, 3:4].flatten()])

    transform_mat = np.identity(4)
    transform_mat[:3, :3] = target_r
    transform_mat[:3, 3] = target_t
    return quaternion_from_matrix(transform_mat)

def toPoseCNNImage(img):
    return cv2.cvtColor(to_np(img).transpose((1,2,0)).astype(np.uint8), cv2.COLOR_BGR2RGB)

class PoseCNNDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data = self.dataset[index]
        img = cv2.cvtColor(to_np(data[0]).transpose((1,2,0)).astype(np.uint8), cv2.COLOR_BGR2RGB)
        depth = to_np(data[1])
        path = '{}/data/{}-meta.mat'.format(self.dataset.dataset_root, self.dataset.getPath(index))
        meta_data = scio.loadmat(path)
        return img, depth, meta_data
    
    def __len__(self):
        return len(self.dataset)

class PoseCNNFeaturizer(object):
    def __init__(self,
                 ):
        data_path = 'data/LOV'
        model_checkpoint = 'data/demo_models/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt'
        cfg_from_file('experiments/cfgs/lov_color_2d.yml')
        cfg.GPU_ID = 1
        device_name = '/gpu:{:d}'.format(1)

        cfg.TRAIN.NUM_STEPS = 1
        cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
        if cfg.NETWORK == 'FCN8VGG':
            raise ValueError(cfg.NETWOK)
            #path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
            #cfg.TRAIN.MODEL_PATH = path
        cfg.TRAIN.TRAINABLE = False
        cfg.TRAIN.VOTING_THRESHOLD = cfg.TEST.VOTING_THRESHOLD

        cfg.RIG = 'data/LOV/camera.json'
        cfg.CAD = 'data/LOV/models.txt'
        cfg.POSE = 'data/LOV/poses.txt'
        cfg.BACKGROUND = 'data/cache/backgrounds.pkl'
        cfg.IS_TRAIN = False

        from networks.factory import get_network
        self.network = get_network('vgg16_convs')
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        saver.restore(self.sess, model_checkpoint)
        self.points_all = getYCBPoints(data_path)[1]
        self.extents = getYCBExtents(data_path)
        self.voxelizer = Voxelizer(cfg.TEST.GRID_SIZE, ycb_num_classes)
        self.voxelizer.setup(-3, -3, -3, 3, 3, 4)
    def __call__(self, im, depth, meta_data):
        seg = feature_net_single_frame(self.sess, self.network, 
                im, depth, meta_data, 
                self.voxelizer, self.extents, self.points_all)
        return seg

