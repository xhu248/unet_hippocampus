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

import os
import torch
from os.path import exists

from configs.Config_chd import get_config
from configs.Config_chd_add import get_add_config
from datasets.chd_dataset.preprocessing import preprocess_data
from datasets.chd_dataset.create_splits import create_splits
from experiments.FCNExperiment import FCNExperiment
from experiments.ChdExperiment import CHDExperiment
from datasets.downsanpling_data import downsampling_image

import datetime
import time

import matplotlib
import matplotlib.pyplot as plt

from datasets.chd_dataset.rearrange_dir import rearrange_dir


def training():

    c = get_config()
    c_addition = get_add_config()

    dataset_name = 'CHD_segmentation_dataset'
    # dataset_name = 'Task04_Hippocampus'
    # dataset_name = 'Task02_Heart'
    # download_dataset(dest_path=c.data_root_dir, dataset=dataset_name, id=c.google_drive_id)
    # c.do_load_checkpoint = True
    # c.checkpoint_dir = c.base_dir + '/20190801-_unet_experiment' + '/checkpoint/checkpoint_current'
    # c.checkpoint_file = "checkpoint_last.pth.tar"

    # rearrange_dir(root_dir=os.path.join(c.data_root_dir, dataset_name))
    if not exists(os.path.join(os.path.join(c.data_root_dir, dataset_name), 'preprocessed')):
        print('Preprocessing data. [STARTED]')
        preprocess_data(root_dir=os.path.join(c.data_root_dir, dataset_name))
        create_splits(output_dir=c.split_dir, image_dir=c.data_dir)
        print('Preprocessing data. [DONE]')
    else:
        print('The data has already been preprocessed. It will not be preprocessed again. Delete the folder to enforce it.')

    preprocess_data(root_dir=os.path.join(c.data_root_dir, dataset_name))
    downsampling_image(c.data_dir, output_dir=c_addition.scaled_image_32_dir)
    # create_splits(output_dir=c.split_dir, image_dir=c.scaled_image_32_dir)
    exp = FCNExperiment(config=c, name='fcn_experiment', n_epochs=c.n_epochs,
                        seed=42, append_rnd_to_name=c.append_rnd_string)   # visdomlogger_kwargs={"auto_start": c.start_visdom}

    exp.run()
    exp.run_test(setup=False)

def testing():

    c = get_config()

    c.do_load_checkpoint = True
    #c.checkpoint_dir = c.base_dir + '/20190424-020641_unet_experiment' + '/checkpoint/checkpoint_current' # dice_cost train
    # c.checkpoint_dir = c.base_dir + '/20190424-234657_unet_experiment' + '/checkpoint/checkpoint_last' # SDG
    c.checkpoint_dir = c.base_dir + '/20190906-085449_unet_experiment' + '/checkpoint/checkpoint_current'
    # c.checkpoint_file = "checkpoint_last.pth.tar"
    # c.cross_vali_index = valiIndex


    cross_vali_result_all_dir = os.path.join(c.base_dir, c.dataset_name
                                             + '_' + str(
        c.batch_size) + c.cross_vali_result_all_dir + datetime.datetime.now().strftime("_%Y%m%d-%H%M%S"))
    if not os.path.exists(cross_vali_result_all_dir):
        os.makedirs(cross_vali_result_all_dir)
        print('Created' + cross_vali_result_all_dir + '...')
        c.base_dir = cross_vali_result_all_dir
        c.cross_vali_result_all_dir = os.path.join(cross_vali_result_all_dir, "results")
        os.makedirs(c.cross_vali_result_all_dir)


    exp = FCNExperiment(config=c, name='chd_test', n_epochs=c.n_epochs,
                               seed=42, globs=globals())
    exp.run_test(setup=True)

def inference():


    c = get_config()
    c.do_load_checkpoint = True
    c.checkpoint_dir = c.base_dir + '/20191010-092447_fcn_experiment' + '/checkpoint/checkpoint_current'
    exp = CHDExperiment(config=c, name='chd_test', n_epochs=c.n_epochs,
                               seed=42, globs=globals())
    exp.run_test(setup=True)
    exp.inference()


if __name__ == "__main__":
    training()

