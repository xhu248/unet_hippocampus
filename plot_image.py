import matplotlib
import matplotlib.pyplot as plt

import os
import numpy as np
import torch
import torch.nn.functional as F
from configs.Config_chd import get_config
from utilities.file_and_folder_operations import subfiles

def reshape_array(numpy_array, axis=1):
    image_shape = numpy_array.shape[1]
    channel = numpy_array.shape[0]
    if axis == 1:
        slice_img = numpy_array[:, 0, :, :].reshape(1, channel, image_shape, image_shape)
        slice_len = np.shape(numpy_array)[1]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, k, :, :].reshape(1, channel, image_shape, image_shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img
    elif axis == 2:
        slice_img = numpy_array[:, :, 0, :].reshape(1, channel, image_shape, image_shape)
        slice_len = np.shape(numpy_array)[2]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, :, k, :].reshape(1, channel, image_shape, image_shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img
    elif axis == 3:
        slice_img = numpy_array[:, :, :, 0].reshape(1, channel, image_shape, image_shape)
        slice_len = np.shape(numpy_array)[3]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, :, :, k].reshape(1, channel, image_shape, image_shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img

if __name__ == '__main__':
    c = get_config()
    """ 
    data_dir = c.data_dir
    image_num = '1016'
    scaled_image_dir = os.path.join(c.data_dir, 'scaled_to_16')
    scaled_image = os.path.join(c.data_dir, 'ct_1083_image.npy')

    train_image = np.load(scaled_image)
    label_image = np.load(scaled_image)[:, 1]

    max_value = label_image.max()
    plt.imshow(train_image[12], cmap='gray')
    plt.show()
    plt.imshow(val_image[12], cmap='gray')
    plt.show()
    
    pred_dir = os.path.join(c.base_dir, c.dataset_name
                                             + '_' + str(
        c.batch_size) + c.cross_vali_result_all_dir + '_20190425-213808')
    

    test_num = c.dataset_name + '_006'
    image_dir = os.path.join(pred_dir, 'results', 'pred_' + test_num + '.npy')


    # all_image = np.load(image_dir)[25]


    plt.figure(1)
    for i in range(np.shape(all_image)[0]):
        plt.subplot(1,3,i+1)
        plt.imshow(train_image[i], cmap='gray')
        if i == 0:
            plt.xlabel('original image')
        elif i == 1:
            plt.xlabel('label image')
        else:
            plt.xlabel('segmented image')

    if not os.path.exists(os.path.join(pred_dir, 'images')):
        os.makedirs(os.path.join(pred_dir, 'images'))

    plt.savefig(os.path.join(pred_dir, 'images') + '/_006_25.jpg')
    plt.show()
    """
    n = 4
    k = 115
    scaled_16_files = subfiles(c.scaled_image_16_dir, suffix='.npy', join=False)
    pred_32_files = subfiles(c.stage_1_dir_32, suffix='64.npy', join=False)
    org_files = subfiles(c.data_dir, suffix='.npy', join=False)

    ############ original image and target ########################
    file = org_files[2]
    data = np.load(os.path.join(c.data_dir, file))
    data = reshape_array(data, axis=3)

    image = data[:, 0]
    target = data[:, 1]

    ############ down scale using interpolation ########################
    data = torch.tensor(data)
    data_256 = F.interpolate(data, scale_factor=1/16, mode='bilinear')
    image_256 = data_256[:, 0]
    target_256 = data_256[:, 1]
    
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('image:%d,  slice:%d, original image' % (n, k))
    plt.imshow(image[k], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('image:%d,  slice:%d, original target' % (n, k))
    plt.imshow(target[k], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('image:%d,  slice:%d, image scale by 0.5' % (n, k))
    plt.imshow(image_256[k], cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title('image:%d,  slice:%d, target scale by 0.5' % (n, k))
    plt.imshow(target_256[k], cmap='gray')
    plt.show()

    ############ down scale using max-pooling ########################
    file_64 = pred_32_files[n]
    pred_64 = np.load(os.path.join(c.stage_1_dir_32, file_64))[:, 0:8]
    target_64 = np.load(os.path.join(c.stage_1_dir_32, file_64))[:, 8:9]

    pred_64 = torch.tensor(pred_64).float()
    target_64 = torch.tensor(target_64).long()

    # 32*32 image and target
    data_32 = np.load(os.path.join(c.scaled_image_32_dir, 'ct_1010_image.npy'))
    image_32 = data_32[:, 0]
    target_32 = data_32[:, 1]

    soft_max = F.softmax(pred_64[k:k + 1], dim=1)
    cf_img = torch.max(soft_max, 1)[0].numpy()
    pred_img = torch.argmax(soft_max, dim=1)

    # plot target
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.title('image:%d,  slice:%d, confidence' % (n, k))
    plt.imshow(cf_img[0], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('image:%d,  slice:%d, target' % (n, k))
    plt.imshow(target_64[k][0], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('image:%d,  slice:%d, pred_image' % (n, k))
    plt.imshow(pred_img[0], cmap='gray')
    plt.show()

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.title('image:%d,  slice:%d, original image' % (n, k))
    plt.imshow(image_32[k], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('image:%d,  slice:%d, original target' % (n, k))
    plt.imshow(target_32[k], cmap='gray')
    plt.show()


