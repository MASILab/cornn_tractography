#!/bin/bash

# Input directory:
# - inference_mni_2mm.trk
# - T1_N4.nii.gz
# - T12mni_0GenericAffine.mat

in_dir=$1

# Move from MNI to T1 space
# - inference.trk

echo "post_trk.sh: Moving tractogram to T1 space..."
scil_apply_transform_to_tractogram.py $in_dir/inference_mni_2mm.trk $in_dir/T1_N4.nii.gz $in_dir/T12mni_0GenericAffine.mat $in_dir/inference.trk --remove_invalid # no --inverse needed per ANTs convention

# Wrap up

echo "post_trk.sh: Done!"
