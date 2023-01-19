#!/bin/bash

# Inputs:
# - T1.nii.gz

in_dir=$1
slant_dir=$2
wml_dir=$3
num_threads=$4

# Set number of threads for OpenMP operations (ANTs)

export OMP_NUM_THREADS=$num_threads

# Get directories

supp_dir=$CORNN_DIR/supplemental
src_dir=$CORNN_DIR/src

# Generate mask:
# - T1_mask.nii.gz

echo "prep_T1.sh: Computing T1 mask..."
cmd="fslmaths $slant_dir/FinalResult/T1_seg.nii.gz -div $slant_dir/FinalResult/T1_seg.nii.gz -fillh $in_dir/T1_mask.nii.gz -odt int"
[ ! -f $in_dir/T1_mask.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Output exists, skipping!"

# Bias correction
# - T1_N4.nii.gz

echo "prep_T1.sh: Bias correcting T1..."
cmd="N4BiasFieldCorrection -d 3 -i $in_dir/T1.nii.gz -x $in_dir/T1_mask.nii.gz -o $in_dir/T1_N4.nii.gz"
[ ! -f $in_dir/T1_N4.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Output exists, skipping!"

# Generate tissue classes
# - T1_5tt.nii.gz

echo "prep_T1.sh: Computing 5tt classes..."
cmd="5ttgen fsl $in_dir/T1_N4.nii.gz $in_dir/T1_5tt.nii.gz -mask $in_dir/T1_mask.nii.gz -nocrop -scratch $in_dir -nthreads $num_threads"
[ ! -f $in_dir/T1_5tt.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Output exists, skipping!"

# Generate seed map:
# - T1_seed.nii.gz

echo "prep_T1.sh: Computing seed mask..."
cmd="fslmaths $in_dir/T1_5tt.nii.gz -roi 0 -1 0 -1 0 -1 2 1 -bin -Tmax $in_dir/T1_seed.nii.gz -odt int"
[ ! -f $in_dir/T1_seed.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Output exists, skipping!"

# Register to MNI template:
# - T12mni_0GenericAffine.mat

echo "prep_T1.sh: Registering to MNI space at 1mm isotropic..."
cmd="antsRegistrationSyN.sh -d 3 -m $in_dir/T1_N4.nii.gz -f $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_1mm.nii.gz -t r -o $in_dir/T12mni_"
[ ! -f $in_dir/T12mni_0GenericAffine.mat ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Transform exists, skipping!"
cmd="mv $in_dir/T12mni_Warped.nii.gz $in_dir/T1_N4_mni_1mm.nii.gz"
[ ! -f $in_dir/T1_N4_mni_1mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Outputs renamed, skipping!"
cmd="rm $in_dir/T12mni_InverseWarped.nii.gz"
[ -f $in_dir/T12mni_InverseWarped.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Outputs cleaned, skipping!"

# Move data to MNI
# - T1_N4_mni_2mm.nii.gz
# - T1_mask_mni_2mm.nii.gz
# - T1_seed_mni_2mm.nii.gz
# - T1_5tt_mni_2mm.nii.gz

echo "prep_T1.sh: Moving images to MNI space at 2mm isotropic..."
cmd="antsApplyTransforms -d 3 -e 0 -r $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_N4.nii.gz   -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_N4_mni_2mm.nii.gz   -n Linear"
[ ! -f $in_dir/T1_N4_mni_2mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: T1_N4 transformed, skipping!"
cmd="antsApplyTransforms -d 3 -e 0 -r $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_mask.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_mask_mni_2mm.nii.gz -n NearestNeighbor"
[ ! -f $in_dir/T1_mask_mni_2mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Mask transformed, skipping!"
cmd="antsApplyTransforms -d 3 -e 0 -r $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_seed.nii.gz -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_seed_mni_2mm.nii.gz -n NearestNeighbor"
[ ! -f $in_dir/T1_seed_mni_2mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: Seeds transformed, skipping!"
cmd="antsApplyTransforms -d 3 -e 3 -r $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_5tt.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_5tt_mni_2mm.nii.gz  -n Linear"
[ ! -f $in_dir/T1_5tt_mni_2mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: 5tt transformed, skipping!"

# Prep SLANT:
# - T1_slant.nii.gz
# - T1_slant_mni_2mm.nii.gz

echo "prep_T1.sh: Preparing SLANT..."
cmd="python $src_dir/prep_slant.py $slant_dir/FinalResult/T1_seg.nii.gz $in_dir/T1_slant.nii.gz"
[ ! -f $in_dir/T1_slant.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: SLANT grouped, skipping!"
cmd="antsApplyTransforms -d 3 -e 3 -r $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_slant.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_slant_mni_2mm.nii.gz -n NearestNeighbor"
[ ! -f $in_dir/T1_slant_mni_2mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: SLANT transformed, skipping!"

# Prep WML:
# - T1_tractseg.nii.gz
# - T1_tractseg_mni_2mm.nii.gz

echo "prep_T1.sh: Preparing WML..."
cmd="fslmerge -t $in_dir/T1_tractseg.nii.gz $wml_dir/orig/*.nii.gz"
[ ! -f $in_dir/T1_tractseg.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: WML merged, skipping!"
cmd="antsApplyTransforms -d 3 -e 3 -r $supp_dir/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz -i $in_dir/T1_tractseg.nii.gz  -t $in_dir/T12mni_0GenericAffine.mat -o $in_dir/T1_tractseg_mni_2mm.nii.gz -n Linear"
[ ! -f $in_dir/T1_tractseg_mni_2mm.nii.gz ] && (echo $cmd && eval $cmd) || echo "prep_T1.sh: WML transformed, skipping!"

# Wrap up

echo "prep_T1.sh: Done!"
