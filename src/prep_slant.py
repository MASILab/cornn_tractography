# Group SLANT Labels
# Leon Cai
# MASI Lab
# November 6, 2022

# Set Up

import os
import sys
import numpy as np
import nibabel as nib

# Variables

CORNN_DIR = os.getenv('CORNN_DIR')
GROUPS_FILE = os.path.join(CORNN_DIR, 'supplemental', 'slant_groupings_12.csv')

# Go!

if __name__ == '__main__':

    seg_file = sys.argv[1]
    out_file = sys.argv[2]
    level = 7 # 6 (32) or 7 (46) seem to be sweet spot (tractseg has 72)
    
    assert level >= 0 and level <= 12, 'Parameter level must be between 0 and 12 inclusive. Aborting.'

    seg_nii = nib.load(seg_file)
    seg_img = seg_nii.get_fdata().astype(int)

    groups_raw = np.genfromtxt(GROUPS_FILE, delimiter=',')[1:, 1:].astype(int)

    labels = groups_raw[:, 0]
    groups = groups_raw[:, level]

    # print('Level {} has {} groups'.format(level, len(np.unique(groups))))

    for i, label in enumerate(labels):
        seg_img[seg_img == label] = groups[i]

    out_img = np.eye(len(np.unique(groups))+1).astype(int)[seg_img][:, :, :, 1:]
    out_nii = nib.Nifti1Image(out_img, seg_nii.affine, dtype=np.uint8)
    nib.save(out_nii, out_file)
