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

from configs.Config_unet import get_config
from datasets.example_dataset.create_splits import create_splits
from datasets.example_dataset.download_dataset import download_dataset
from datasets.example_dataset.preprocessing import preprocess_data
from experiments.UNetExperiment import UNetExperiment
from datasets.downsanpling_data import downsampling_image

import datetime
import time

import matplotlib
import matplotlib.pyplot as plt


def training():

    c = get_config()

    dataset_name = 'Task04_Hippocampus'
    # dataset_name = 'Task02_Heart'
    # download_dataset(dest_path=c.data_root_dir, dataset=dataset_name, id=c.google_drive_id)
    # c.do_load_checkpoint = True
    # c.checkpoint_dir = c.base_dir + '/20190801-_unet_experiment' + '/checkpoint/checkpoint_current'
    # c.checkpoint_file = "checkpoint_last.pth.tar"

    if not exists(os.path.join(os.path.join(c.data_root_dir, dataset_name), 'preprocessed')):
        print('Preprocessing data. [STARTED]')
        preprocess_data(root_dir=os.path.join(c.data_root_dir, dataset_name))
        create_splits(output_dir=c.split_dir, image_dir=c.data_dir)
        print('Preprocessing data. [DONE]')
    else:
        print('The data has already been preprocessed. It will not be preprocessed again. Delete the folder to enforce it.')


    exp = UNetExperiment(config=c, name='unet_experiment', n_epochs=c.n_epochs,
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


    exp = UNetExperiment(config=c, name='unet_test', n_epochs=c.n_epochs,
                               seed=42, globs=globals())
    exp.run_test(setup=True)

if __name__ == "__main__":
    training()

