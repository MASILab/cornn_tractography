#!/bin/bash

# Inputs:
# - T1_gen_mni_2mm
# - T1_N4.nii.gz (prep_t1.sh)
# - T12mni_0GenericAffine.mat (prep_t1.sh)

in_dir=$1

# Move from MNI to T1 space

echo "post_trk.sh: Moving tractogram to T1 space..."
scil_apply_transform_to_tractogram.py $in_dir/T1_gen_mni_2mm.trk $in_dir/T1_N4.nii.gz $in_dir/T12mni_0GenericAffine.mat $in_dir/T1_gen.trk --remove_invalid # no --inverse needed per ANTs convention

# Wrap up

echo "post_trk.sh: Done!"
