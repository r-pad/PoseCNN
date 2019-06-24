from .test import *
from .test import _get_image_blob

ycb_num_classes = 22

ycb_symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
ycb_classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
               '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
               '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
               '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')

def getYCBExtents(ycb_path): 
    extent_file = os.path.join(ycb_path, 'extents.txt')
    assert os.path.exists(extent_file), \
        'Path does not exist: {}'.format(extent_file)

    extents = np.zeros((ycb_num_classes, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_file)

    return extents

def getYCBPoints(ycb_path):
    points = [[] for _ in xrange(len(ycb_classes))]
    num = np.inf

    for i in xrange(1, len(ycb_classes)):
        point_file = os.path.join(ycb_path, 'models', ycb_classes[i], 'points.xyz')
        #print point_file
        assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    points_all = np.zeros((ycb_num_classes, num, 3), dtype=np.float32)
    for i in xrange(1, len(ycb_classes)):
        points_all[i, :, :] = points[i][:num, :]

    return points, points_all

def im_feature_single_frame(sess, net, im, im_depth, meta_data, voxelizer, extents, points, symmetry, num_classes):
    """segment image
    """

    # compute image blob
    im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = _get_image_blob(im, im_depth, meta_data)
    im_scale = im_scale_factors[0]
    # construct the meta data
    """
    format of the meta_data
    intrinsic matrix: meta_data[0 ~ 8]
    inverse intrinsic matrix: meta_data[9 ~ 17]
    pose_world2live: meta_data[18 ~ 29]
    pose_live2world: meta_data[30 ~ 41]
    voxel step size: meta_data[42, 43, 44]
    voxel min value: meta_data[45, 46, 47]
    """
    K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros(48, dtype=np.float32)
    mdata[0:9] = K.flatten()
    mdata[9:18] = Kinv.flatten()
    # mdata[18:30] = pose_world2live.flatten()
    # mdata[30:42] = pose_live2world.flatten()
    mdata[42] = voxelizer.step_x
    mdata[43] = voxelizer.step_y
    mdata[44] = voxelizer.step_z
    mdata[45] = voxelizer.min_x
    mdata[46] = voxelizer.min_y
    mdata[47] = voxelizer.min_z
    if cfg.FLIP_X:
        mdata[0] = -1 * mdata[0]
        mdata[9] = -1 * mdata[9]
        mdata[11] = -1 * mdata[11]
    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    meta_data_blob[0,0,0,:] = mdata

    # use a fake label blob of ones
    height = int(im_depth.shape[0] * im_scale)
    width = int(im_depth.shape[1] * im_scale)
    label_blob = np.ones((1, height, width), dtype=np.int32)

    pose_blob = np.zeros((1, 13), dtype=np.float32)
    vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    if cfg.INPUT == 'RGBD':
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
    else:
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)
    if cfg.NETWORK == 'FCN8VGG':
        labels_2d, probs = sess.run([net.label_2d, net.prob], feed_dict=feed_dict)
    else:
        if cfg.TEST.VERTEX_REG_2D:
            if cfg.TEST.POSE_REG:
                labels_2d, probs, vertex_pred, rois, poses_init, poses_pred, feats, fc6 = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                              net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh'), \
                              net.get_output('pool_score'), net.get_output('fc6')])

                # non-maximum suppression
                keep = nms(rois, 0.5)
                rois = rois[keep, :]
                poses_init = poses_init[keep, :]
                poses_pred = poses_pred[keep, :]
                feats = feats[keep, :]
                fc6 = fc6[keep, :]
                #print keep
                #print rois

                # combine poses
                num = rois.shape[0]
                poses = poses_init
                for i in xrange(num):
                    class_id = int(rois[i, 1])
                    if class_id >= 0:
                        poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            else:
                labels_2d, probs, vertex_pred, rois, poses = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
                #print rois
                #print rois.shape
                # non-maximum suppression
                # keep = nms(rois[:, 2:], 0.5)
                # rois = rois[keep, :]
                # poses = poses[keep, :]

                #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
                #vertex_pred = []
                #rois = []
                #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        elif cfg.TEST.VERTEX_REG_3D:
            labels_2d, probs, vertex_pred = \
                sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred')])
            rois = []
            poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            vertex_pred = []
            rois = []
            poses = []

    return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses, feats, fc6

def feature_net_single_frame(sess, net, im, depth, meta_data, 
        voxelizer, ycb_extents, ycb_points_all):
    im = pad_im(im, 16)
    depth = pad_im(depth, 16)
    labels, probs, vertex_pred, rois, poses, feats, fc6  = im_feature_single_frame(sess, net, \
            im, depth, meta_data, voxelizer, \
            ycb_extents, ycb_points_all, ycb_symmetry, ycb_num_classes)

    labels = unpad_im(labels, 16)
    im_scale = cfg.TEST.SCALES_BASE[0]
    labels_new = cv2.resize(labels, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_NEAREST)
       
    seg = {'labels': labels_new, 'rois': rois, 'poses': poses, 'feats':feats, 'fc6':fc6} 
    return seg
