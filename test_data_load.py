from collections import defaultdict

from medpy.io import load
import os
import numpy as np

from datasets.utils import reshape
from utilities.file_and_folder_operations import subfiles

def testdata_preprocess(input_dir, output_dir):

    nii_files = subfiles(input_dir, suffix=".nii.gz", join=False)

    for i in range(0, len(nii_files)):
        if nii_files[i].startswith("._"):
            nii_files[i] = nii_files[i][2:]

    for f in nii_files:
        image, a = load(os.path.join(input_dir, f))  # ??? what's the output-- image_header?
        print(f)

        image = (image - image.min()) / (image.max() - image.min())

        image = reshape(image, append_value=0, new_shape=(64, 64, 64))

        np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), image)

if __name__ == "__main__":
    data_root_dir = os.path.abspath('data')
    test_dir = os.path.join(data_root_dir, 'Task04_Hippocampus/imagesTs')
    test_processed_dir = os.path.join(test_dir, 'processed')

    if not os.path.exists(test_processed_dir):
        os.makedirs(test_processed_dir)
        print('Create' + test_processed_dir + '...')

    testdata_preprocess(test_dir, test_processed_dir)