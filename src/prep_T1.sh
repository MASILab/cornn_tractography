#!/bin/bash

# Inputs:
# - T1.nii.gz

in_dir=$1
atlas_dir=$2
slant_dir=$3
wml_dir=$4

# Generate mask:
# - T1_mask.nii.gz

echo "prep_T1.sh: Computing T1 mask..."
fslmaths $in_dir/T1_seg.nii.gz -div $in_dir/T1_seg.nii.gz -fillh $in_dir/T1_mask.nii.gz -odt int

# Bias correction
# - T1_N4.nii.gz

echo "prep_T1.sh: Bias correcting T1..."
N4BiasFieldCorrection -d 3 -i $in_dir/T1.nii.gz -x $in_dir/T1_mask.nii.gz -o $in_dir/T1_N4.nii.gz

# Generate tissue classes
# - T1_5tt.nii.gz

echo "prep_T1.sh: Computing 5tt classes..."
5ttgen fsl $in_dir/T1_N4.nii.gz $in_dir/T1_5tt.nii.gz -mask $in_dir/T1_mask.nii.gz -nocrop

# Generate seed map:
# - T1_seed.nii.gz

echo "prep_T1.sh: Computing seed mask..."
fslmaths $in_dir/T1_5tt.nii.gz -roi 0 -1 0 -1 0 -1 2 1 -bin -Tmax $in_dir/T1_seed.nii.gz -odt int

# Register to MNI template:
# - T12mni_0GenericAffine.mat

echo "prep_T1.sh: Registering to MNI space at 1mm isotropic..."
antsRegistrationSyN.sh -d 3 -m $in_dir/T1_N4.nii.gz -f $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -t r -o $in_dir/T12mni_
mv $in_dir/T12mni_Warped.nii.gz $in_dir/T1_N4_mni_1mm.nii.gz
rm $in_dir/T12mni_InverseWarped.nii.gz

# Move data to MNI
# - T1_N4_mni_2mm.nii.gz
# - T1_mask_mni_2mm.nii.gz
# - T1_seed_mni_2mm.nii.gz
# - T1_5tt_mni_2mm.nii.gz

echo "prep_T1.sh: Moving images to MNI space at 2mm isotropic..."
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_N4.nii.gz   -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_N4_mni_2mm.nii.gz   -n Linear
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_mask.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_mask_mni_2mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_seed.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seed_mni_2mm.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_5tt.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_5tt_mni_2mm.nii.gz  -n Linear

# Prep SLANT:
# - T1_slant.nii.gz
# - T1_slant_mni_2mm.nii.gz

if [ -d $slant_dir ]
then
    echo "prep_T1.sh: Preparing SLANT..."
    python group_slant.py $slant_dir/FinalResult/T1_seg.nii.gz $in_dir/T1_slant.nii.gz # T1_slant is one-hot encoded
    antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_slant.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_slant_mni_2mm.nii.gz -n NearestNeighbor
fi

# Prep WML:
# - T1_tractseg.nii.gz
# - T1_tractseg_mni_2mm.nii.gz

if [ -d $wml_dir ]
then
    echo "prep_T1.sh: Preparing WML..."
    fslmerge -t $in_dir/T1_tractseg.nii.gz $wml_dir/orig/*.nii.gz
    antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_tractseg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_tractseg_mni_2mm.nii.gz -n Linear
if

# Wrap up

echo "prep_T1.sh: Done!"
