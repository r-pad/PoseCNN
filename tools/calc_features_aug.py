import _init_paths
import argparse
import os
import random
import time
import numpy as np
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.image_processing import ColorJitter, ImageNormalizer
from object_pose_utils.datasets.ycb_occlusion_augmentation import YCBOcclusionAugmentor
from object_pose_utils.datasets.point_processing import PointShifter
from object_pose_utils.utils import to_np

from tqdm import tqdm, trange


from time import sleep
import contextlib
import sys

from featurization import PoseCNNFeaturizer, toPoseCNNImage, getObjectGTQuaternion
import torch
import scipy.io as scio 

import os
import sys
module_path = os.path.abspath(os.path.join('tools'))
if module_path not in sys.path:
        sys.path.append(module_path)
        module_path = os.path.abspath(os.path.join('lib'))
        if module_path not in sys.path:
                sys.path.append(module_path)

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default =  'datasets/ycb/YCB_Video_Dataset', 
        help='Dataset root dir (''YCB_Video_Dataset'')')
parser.add_argument('--dataset_mode', type=str, default = 'train_syn_valid',
        help='Dataset mode')
parser.add_argument('--num_augmentations', type=int, default = 0, 
        help='Number of augmented images per render')
parser.add_argument('--workers', type=int, default = 10, help='Number of data loading workers')
#parser.add_argument('--weights', type=str, help='PoseNetGlobal weights file')
parser.add_argument('--output_folder', type=str, help='Feature save location')
parser.add_argument('--object_indices', type=int, nargs='+', default = None, help='Object indices to featureize')
parser.add_argument('--start_index', type=int, default = 0, help='Starting augmentation index')
opt = parser.parse_args()

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    num_points = 1000 #number of points on the input pointcloud
    num_objects = 21
    if(opt.object_indices is None):
        opt.object_indices = list(range(1,num_objects+1))
    estimator = PoseCNNFeaturizer()
    
    output_format = [otypes.IMAGE, 
                     otypes.DEPTH_IMAGE]
   
    with std_out_err_redirect_tqdm() as orig_stdout:
        preprocessors = []
        postprocessors = []
        if(opt.num_augmentations > 0):
            preprocessors.extend([YCBOcclusionAugmentor(opt.dataset_root), 
                                  ColorJitter(),])
            postprocessors.append(PointShifter())
        
        dataset = YCBDataset(opt.dataset_root, mode = opt.dataset_mode,
                             object_list = opt.object_indices, 
                             output_data = output_format,
                             resample_on_error = False,
                             preprocessors = preprocessors, 
                             postprocessors = postprocessors,
                             image_size = [640, 480], num_points=1000)
        _, u_idxs = np.unique(zip(*dataset.image_list)[0], return_index = True)
        dataset.image_list = np.array(dataset.image_list)[u_idxs].tolist()
        dataset.list_obj = np.array(dataset.list_obj)[u_idxs].tolist()

        classes = dataset.classes
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
        #pbar.set_description('Featurizing {}'.format(classes[cls]))

        if(opt.num_augmentations > 0):
            pbar_aug = trange(opt.start_index, opt.num_augmentations, file=orig_stdout, dynamic_ncols=True)
        else:
            pbar_aug = [None]
        for aug_idx in pbar_aug:
            
            pbar_save = tqdm(enumerate(dataloader), total = len(dataloader),
                             file=orig_stdout, dynamic_ncols=True)
            for i, data in pbar_save:
                if(len(data) == 0 or len(data[0]) == 0):
                    continue
                img, depth = data
                img = toPoseCNNImage(img[0])
                depth = to_np(depth[0])
                data_path = dataset.image_list[i]
                
                path = '{}/data/{}-meta.mat'.format(dataset.dataset_root, dataset.getPath(i))
                meta_data = scio.loadmat(path)
                try:
                    seg = estimator(img, depth, meta_data)
                except Exception as e:
                    print(e)
                    continue 
                for pose_idx, cls in enumerate(seg['rois'][:,1]):
                    cls = int(cls)
                    quat = getObjectGTQuaternion(meta_data, cls)
                    feat = seg['feats'][pose_idx]
                    fc6 = seg['fc6'][pose_idx]
                    if(opt.num_augmentations > 0):
                        output_filename = '{0}/data/{1}_{2}_{3}_feat.npz'.format(opt.output_folder, 
                                data_path[0], classes[cls], aug_idx)
                    else:
                        output_filename = '{0}/data/{1}_{2}_feat.npz'.format(opt.output_folder, 
                                data_path[0], classes[cls])

                    #pbar_save.set_description(output_filename)
                    if not os.path.exists(os.path.dirname(output_filename)):
                        os.makedirs(os.path.dirname(output_filename))
                    np.savez(output_filename, quat = quat, feat = feat, fc6 = fc6)
                    
           
if __name__ == '__main__':
    main()
