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
import pickle
import cv2

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.RecursiveUNet import UNet
from models.fcn8s import FCN8s
from models.fcn32s import FCN32s
from models.fcn8s import FCN8s
from loss_functions.dice_loss import SoftDiceLoss

from loss_functions.metrics import dice_pytorch

from utilities.file_and_folder_operations import subfiles

import matplotlib.pyplot as plt
from loss_functions.metrics import dice_pytorch


class CHDExperiment(PytorchExperiment):
    """
    """

    def setup(self):

        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']
        keys = tr_keys + val_keys
        test_keys = splits[self.config.fold]['test']

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')    #

        self.model = UNet(num_classes=self.config.num_classes, num_downs=3)

        self.model.to(self.device)

        self.data_loader = NumpyDataSet(self.config.data_dir, target_size=256, batch_size=self.config.batch_size,
                                        keys=keys, mode='test', do_reshuffle=False)

        self.data_16_loader = NumpyDataSet(self.config.scaled_image_32_dir, target_size=32, batch_size=self.config.batch_size,
                                        keys=keys, mode='test', do_reshuffle=False)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax für DICE Loss!

        # weight = torch.tensor([1, 30, 30]).float().to(self.device)
        self.ce_loss = torch.nn.CrossEntropyLoss()  # Kein Softmax für CE Loss -> ist in torch schon mit drin!
        # self.dice_pytorch = dice_pytorch(self.config.num_classes)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))



    def inference(self):
        self.elog.print('=====INFERENCE=====')
        image_files = subfiles(self.config.scaled_image_32_dir, suffix='.npy')

        with torch.no_grad():
            if os.path.exists(self.config.stage_1_dir_32):
               print('stage_1_dir already exists')
            else:
                for data_batch in self.data_16_loader:
                    file_dir = data_batch['fnames']
                    data_16 = data_batch['data'][0].float().to(self.device)  # size (8, 1, 16, 16)
                    target_16 = data_batch['seg'][0].float().to(self.device)

                    if not os.path.exists(self.config.stage_1_dir_32):
                        os.mkdir(self.config.stage_1_dir_32)
                        print('Creatting stage_1_dir...')

                    pred_16 = self.model(data_16)
                    pred_16_softmax = F.softmax(pred_16, dim=1)
                    dice_16 = 1 - self.dice_loss(pred_16_softmax, target_16.squeeze())
                    ce_16 = self.ce_loss(pred_16, target_16.squeeze().long())


                    if dice_16 < 0.6:
                        print(file_dir[0])
                        print(data_batch['slice_idxs'])

                    pred_32 = F.interpolate(pred_16, scale_factor=2, mode='bilinear')
                    target_32 = F.interpolate(target_16, scale_factor=2, mode='bilinear')
                    pred_32_softmax = F.softmax(pred_32, dim=1)
                    dice_32 = 1 - self.dice_loss(pred_32_softmax, target_32.squeeze())
                    ce_32 = self.ce_loss(pred_32, target_32.squeeze().long())

                    # print('dice_16: %.4f  dice_32: %.4f' % (dice_16, dice_32))
                    print('dice_16: %.4f dice_32: %.4f ce_16: %.4f  ce_32: %.4f' % (dice_16, dice_32, ce_16, ce_32))

                    for k in range(self.config.batch_size):
                        filename = file_dir[k][0][-14:-4]
                        output_dir = os.path.join(self.config.stage_1_dir_32,
                                                  'pred_' + filename + '_64')
                        if os.path.exists(output_dir + '.npy'):
                            all_data = np.load(output_dir + '.npy')
                            new_data = np.concatenate((pred_32[k:k + 1], target_32[k:k + 1]),
                                                      axis=1)  # size (1,9,32,16)
                            all_data = np.concatenate((all_data, new_data), axis=0)
                        else:
                            all_data = np.concatenate((pred_32[k:k + 1], target_32[k:k + 1]), axis=1)
                            print(filename)

                        np.save(output_dir, all_data)

        # do softmax analysis, and divide the pred image into 4 parts
            pred_32_files = subfiles(self.config.stage_1_dir_32, suffix='32.npy', join=False)

            with torch.no_grad():
                softmax = []
                dice_score = []
                for file in pred_32_files:
                    dice_score = []
                    pred_32 = np.load(os.path.join(self.config.stage_1_dir_32, file))[:, 0:8]  # size (N,8,32,32)
                    target_32 = np.load(os.path.join(self.config.stage_1_dir_32, file))[:, 8:9]

                    pred_32 = torch.tensor(pred_32).float()
                    target_32 = torch.tensor(target_32).long()


                    md_softmax, index, weak_image = softmax_analysis(pred_32, threshold=0)

                    softmax = softmax + md_softmax

                    shape = pred_32.shape
                    image_num = shape[0]

                    for k in range(image_num):
                        pred_softmax = F.softmax(pred_32[k:k+1], 1)
                        dice = self.dice_loss(pred_softmax, target_32[k])
                        pred_image = torch.argmax(pred_softmax, dim=1)
                        dice_score.append(dice)
                    # visualize dice
                    dice_score = np.array(dice_score)
                    avg_dice = np.average(dice_score)
                    min_dice = min(dice_score)
                    # print(file, 'dice_loss:%.4f  min_dice:%.4f' % (avg_dice, min_dice))

                    plot_bar(softmax)

    def compare_dice(self):
        with torch.no_grad():
            for data_batch in self.data_loader:
                file = data_batch['fname']
                data = data_batch['data'][0].float().to(self.device)  # size (8, 1, 256, 256)
                target = data_batch['seg'][0].long().to(self.device)

                file_16 = os.path.join(self.config.scaled_image_32_dir, file[0])
                image_16_tensor = torch.tensor(np.load(file_16))  # size (N, 2, 16, 16)
                data_16 = image_16_tensor[:, 0:1].float().to(self.device)  # size (N, 1, 16, 16)
                target_16 = image_16_tensor[:, 1:2].float().to(self.device)

                idxs = data_batch['slice_indxs']
                start_idx = min(idxs)
                end_idx = max(idxs)

                scaled_data = data_16[start_idx:end_idx + 1]
                scaled_target = target_16[start_idx:end_idx + 1]  # size (8, 8, 16, 16)

                scaled_pred = self.model(scaled_data)
                scaled_pred_softmax = F.softmax(scaled_pred)
                loss_16 = self.dice_loss(scaled_pred_softmax, scaled_target.squeeze())

                upsample_pred = F.interpolate(scaled_target, scale_factor=16, mode='bilinear')
                pred_sofmax = F.softmax(upsample_pred)
                loss = self.dice_loss(pred_sofmax, target_16.squeeze())

                print('loss_16: %.4f  loss: %.4f' % (loss_16, loss))




# input: pred_softmax [8,8,16,16],
# output: softmax with the same
def softmax_analysis(pred, threshold=0.2):
    shape = pred.shape
    image_num = shape[0]

    d = int(shape[3]/2)

    avg_softmax = []
    min_softmax = []
    md_softmax = []
    index = []
    weak_image = []

    for k in range(image_num):
        image = pred[k:k+1]
        sect_1 = image[:, :, 0:d, 0:d] # size (1, 8, 16, 16)
        sect_2 = image[:, :, 0:d, d:2*d]
        sect_3 = image[:, :, d:2*d, 0:d]
        sect_4 = image[:, :, d:2*d, d:2*d]

        sm_1 = torch.max(F.softmax(sect_1), 1)[0].numpy() # size (1,16,16)
        sm_2 = torch.max(F.softmax(sect_2), 1)[0].numpy()
        sm_3 = torch.max(F.softmax(sect_3), 1)[0].numpy()
        sm_4 = torch.max(F.softmax(sect_4), 1)[0].numpy()

        plt.figure(k)
        plt.subplot(2, 2, 1)
        plt.imshow(sm_1[0], cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(sm_2[0], cmap='gray')
        plt.subplot(2, 2, 3)
        plt.imshow(sm_3[0], cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(sm_4[0], cmap='gray')
        plt.show()


        # do average
        sm_1_avg = np.average(sm_1.squeeze())
        md_softmax.append(np.median(sm_1))
        min_softmax.append(sm_1.min())
        sm_2_avg = np.average(sm_2.squeeze())
        avg_softmax.append(sm_2_avg)
        md_softmax.append(np.median(sm_2))
        sm_3_avg = np.average(sm_3.squeeze())
        avg_softmax.append(sm_3_avg)
        md_softmax.append(np.median(sm_3))
        sm_4_avg = np.average(sm_4.squeeze())
        avg_softmax.append(sm_4_avg)
        md_softmax.append(np.median(sm_4))

        if abs(sm_1_avg - 0.5) < threshold:
            index.append((k, 1))
            weak_image.append(sect_1)

        if abs(sm_2_avg - 0.5) < threshold:
            index.append((k, 2))
            weak_image.append(sect_2)

        if abs(sm_3_avg - 0.5) < threshold:
            index.append((k, 3))
            weak_image.append(sect_3)

        if abs(sm_4_avg - 0.5) < threshold:
            index.append((k, 4))
            weak_image.append(sect_4)

    weak_image = np.array(weak_image)
    return md_softmax, index, weak_image


def compute_dice(pred, target):
    shape = pred.shape
    image_num = shape[0]


def plot_bar(data):
    _ = plt.hist(data, bins='auto')
    plt.show()


