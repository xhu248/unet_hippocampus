from utilities.file_and_folder_operations import subfiles
import numpy as np
from configs.Config_unet import get_config
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def reshape_array(numpy_array, axis=1):
    shape = numpy_array.shape[1]
    if axis == 1:
        slice_img = numpy_array[:, 0, :, :].reshape(1, 2, shape, shape)
        slice_len = np.shape(numpy_array)[1]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, k, :, :].reshape(1, 2, shape, shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img
    elif axis == 2:
        slice_img = numpy_array[:, :, 0, :].reshape(1, 2, shape, shape)
        slice_len = np.shape(numpy_array)[2]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, :, k, :].reshape(1, 2, shape, shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img
    elif axis == 3:
        slice_img = numpy_array[:, :, :, 0].reshape(1, 2, shape, shape)
        slice_len = np.shape(numpy_array)[3]
        for k in range(1, slice_len):
            slice_array = numpy_array[:, :, :, k].reshape(1, 2, shape, shape)
            slice_img = np.concatenate((slice_img, slice_array))
        return slice_img

def downsampling_image():
    project_dir = "/Users/lucasforever24/Downloads/basic_unet_example-master"
    data_path = os.path.join(project_dir, "data/Task04_Hippocampus/preprocessed")

    output_dir = os.path.join(project_dir, "data/Task04_Hippocampus/scaled_to_16")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    # data_path = os.path.join(project_dir, "data/Task01_BrainTumour/preprocessed")
    # image_dir = os.path.join(c.data_dir)
    npy_files = subfiles(data_path, suffix=".npy", join=False)
    for file in npy_files:
        np_path = os.path.join(data_path, file)
        numpy_array = np.load(np_path)

        slice_img = reshape_array(numpy_array)
        slice_data = torch.from_numpy(slice_img)
        pooling_1_data = F.max_pool2d(slice_data, kernel_size=2, stride=2)
        pooling_2_data = F.max_pool2d(pooling_1_data, kernel_size=2, stride=2)

        pooling_2_array = pooling_2_data.numpy()
        np.save(os.path.join(output_dir, file.split('.')[0] + '.npy'), pooling_2_array)
        print(file)


""""
file_num = len(npy_files)
for i in range(1, 50):
    np_path = os.path.join(data_path, npy_files[i])
    numpy_array = np.load(np_path)
    slice_data = reshape_array(numpy_array)
    slice_img = np.concatenate((slice_img, slice_data))

print(np.shape(slice_img))




pooling_1_data = F.max_pool2d(batch_data, kernel_size=2, stride=2)
pooling_2_data = F.max_pool2d(pooling_1_data, kernel_size=2, stride=2)

batch_image = batch_data[150]
plt.figure(1)
plt.subplot(3, 2, 1)
plt.imshow(batch_image[0], cmap='gray')
plt.subplot(3, 2, 2)
plt.imshow(batch_image[1], cmap='gray')


pooling_image = pooling_1_data[150]
plt.figure(1)
plt.subplot(3, 2, 3)
plt.imshow(pooling_image[0], cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(pooling_image[1], cmap='gray')

pooling_image = pooling_2_data[150]
plt.figure(1)
plt.subplot(3, 2, 5)
plt.imshow(pooling_image[0], cmap='gray')
plt.subplot(3, 2, 6)
plt.imshow(pooling_image[1], cmap='gray')

plt.show()

"""