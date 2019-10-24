#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from medpy.io import load
import os
import numpy as np

from datasets.utils import reshape
from utilities.file_and_folder_operations import subfiles

def reshape_array(numpy_array):
    shape = numpy_array.shape[1]
    slice_img = numpy_array[:, :, :, 0].reshape(1, 2, shape, shape)
    slice_len = np.shape(numpy_array)[3]
    for k in range(1, slice_len):
        slice_array = numpy_array[:, :, :, k].reshape(1, 2, shape, shape)
        slice_img = np.concatenate((slice_img, slice_array))

    return slice_img


def preprocess_data(root_dir):
    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')
    output_dir = os.path.join(root_dir, 'preprocessed')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0
    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    for f in nii_files:
        file_dir = os.path.join(output_dir, f.split('.')[0]+'.npy')
        if not os.path.exists(file_dir) and '081' not in f:
            image, _ = load(os.path.join(image_dir, f))
            label, _ = load(os.path.join(label_dir, f.replace('image', 'label')))


            # normalize images
            image = (image - image.min()) / (image.max() - image.min())

            # image = reshape(image, append_value=0, new_shape=(64, 64, 64))
            # label = reshape(label, append_value=0, new_shape=(64, 64, 64))

            result = np.stack((image, label))
            result = reshape_array(result)

            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
            print(f)

    print(total)

