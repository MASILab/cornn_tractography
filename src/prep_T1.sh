#!/bin/bash

# Inputs:
# - T1.nii.gz

in_dir=$1
atlas_dir=$2
src_dir=$3
slant_dir=$4
wml_dir=$5

# Generate mask:
# - T1_mask.nii.gz

echo "prep_T1.sh: Computing T1 mask..."
if [ ! -f $in_dir/T1_mask.nii.gz ]
then
    fslmaths $slant_dir/FinalResult/T1_seg.nii.gz -div $slant_dir/FinalResult/T1_seg.nii.gz -fillh $in_dir/T1_mask.nii.gz -odt int
fi

# Bias correction
# - T1_N4.nii.gz

echo "prep_T1.sh: Bias correcting T1..."
if [ ! -f $in_dir/T1_N4.nii.gz ]
then
N4BiasFieldCorrection -d 3 -i $in_dir/T1.nii.gz -x $in_dir/T1_mask.nii.gz -o $in_dir/T1_N4.nii.gz
fi

# Generate tissue classes
# - T1_5tt.nii.gz

echo "prep_T1.sh: Computing 5tt classes..."
if [ ! -f $in_dir/T1_5tt.nii.gz ]
then
5ttgen fsl $in_dir/T1_N4.nii.gz $in_dir/T1_5tt.nii.gz -mask $in_dir/T1_mask.nii.gz -nocrop
fi

# Generate seed map:
# - T1_seed.nii.gz

echo "prep_T1.sh: Computing seed mask..."
if [ ! -f $in_dir/T1_seed.nii.gz ]
then
fslmaths $in_dir/T1_5tt.nii.gz -roi 0 -1 0 -1 0 -1 2 1 -bin -Tmax $in_dir/T1_seed.nii.gz -odt int
fi

# Register to MNI template:
# - T12mni_0GenericAffine.mat

echo "prep_T1.sh: Registering to MNI space at 1mm isotropic..."
if [ ! -f $in_dir/T12mni_0GenericAffine.mat ]
then
antsRegistrationSyN.sh -d 3 -m $in_dir/T1_N4.nii.gz -f $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -t r -o $in_dir/T12mni_
fi
if [ ! -f $in_dir/T1_N4_mni_1mm.nii.gz ]
then
mv $in_dir/T12mni_Warped.nii.gz $in_dir/T1_N4_mni_1mm.nii.gz
fi
if [ -f $in_dir/T12mni_InverseWarped.nii.gz ]
then
rm $in_dir/T12mni_InverseWarped.nii.gz
fi

# Move data to MNI
# - T1_N4_mni_2mm.nii.gz
# - T1_mask_mni_2mm.nii.gz
# - T1_seed_mni_2mm.nii.gz
# - T1_5tt_mni_2mm.nii.gz

echo "prep_T1.sh: Moving images to MNI space at 2mm isotropic..."
if [ ! -f $in_dir/T1_N4_mni_2mm.nii.gz ]
then
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_N4.nii.gz   -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_N4_mni_2mm.nii.gz   -n Linear
fi
if [ ! -f $in_dir/T1_mask_mni_2mm.nii.gz ]
then
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_mask.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_mask_mni_2mm.nii.gz -n NearestNeighbor
fi
if [ ! -f $in_dir/T1_seed_mni_2mm.nii.gz ]
then
antsApplyTransforms -d 3 -e 0 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_seed.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seed_mni_2mm.nii.gz -n NearestNeighbor
fi
if [ ! -f $in_dir/T1_5tt_mni_2mm.nii.gz ]
then
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_5tt.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_5tt_mni_2mm.nii.gz  -n Linear
fi

# Prep SLANT:
# - T1_slant.nii.gz
# - T1_slant_mni_2mm.nii.gz

echo "prep_T1.sh: Preparing SLANT..."
if [ ! -f $in_dir/T1_slant.nii.gz ]
then
python $src_dir/group_slant.py $slant_dir/FinalResult/T1_seg.nii.gz $in_dir/T1_slant.nii.gz # T1_slant is one-hot encoded
fi
if [ ! -f $in_dir/T1_slant_mni_2mm.nii.gz ]
then
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_slant.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_slant_mni_2mm.nii.gz -n NearestNeighbor
fi

# Prep WML:
# - T1_tractseg.nii.gz
# - T1_tractseg_mni_2mm.nii.gz

echo "prep_T1.sh: Preparing WML..."
if [ ! -f $in_dir/T1_tractseg.nii.gz ]
then
fslmerge -t $in_dir/T1_tractseg.nii.gz $wml_dir/orig/*.nii.gz
fi
if [ ! -f $in_dir/T1_tractseg_mni_2mm.nii.gz ]
then
antsApplyTransforms -d 3 -e 3 -r $atlas_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_tractseg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_tractseg_mni_2mm.nii.gz -n Linear
fi

# Wrap up

echo "prep_T1.sh: Done!"
