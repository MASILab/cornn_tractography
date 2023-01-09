#!/bin/bash

# Inputs:
# - dwmri{.nii.gz/.bval/.bvec} (PreQual)
# - T1_{N4/mask/seed/5tt/N4_mni_2mm}.nii.gz (prep_t1.sh)

in_dir=$1

# Get average b0 for registration
# - dwmri_b0.nii.gz

echo "prep_dwmri.sh: Computing average b0..."
dwiextract $in_dir/dwmri.nii.gz -fslgrad $in_dir/dwmri.bvec $in_dir/dwmri.bval - -bzero | mrmath - mean $in_dir/dwmri_b0.nii.gz -axis 3

# Rigidly register T1 to b0
# - dwmri2T1_0GenericAffine.mat

echo "prep_dwmri.sh: Rigidly registering b0 to T1 space..."
antsRegistrationSyN.sh -d 3 -m $in_dir/dwmri_b0.nii.gz -f $in_dir/T1_N4.nii.gz -t r -o $in_dir/dwmri2T1_
rm $in_dir/dwmri2T1_Warped.nii.gz $in_dir/dwmri2T1_InverseWarped.nii.gz

# Move brain and seed mask to b0 space
# - dwmri_mask.nii.gz
# - dwmri_seed.nii.gz
# - dwmri_5tt.nii.gz

echo "prep_dwmri.sh: Moving masks to diffusion space..."
antsApplyTransforms -d 3 -e 0 -r $in_dir/dwmri_b0.nii.gz -i $in_dir/T1_mask.nii.gz -t [$in_dir/dwmri2T1_0GenericAffine.mat,1] -o $in_dir/dwmri_mask.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 0 -r $in_dir/dwmri_b0.nii.gz -i $in_dir/T1_seed.nii.gz -t [$in_dir/dwmri2T1_0GenericAffine.mat,1] -o $in_dir/dwmri_seed.nii.gz -n NearestNeighbor
antsApplyTransforms -d 3 -e 3 -r $in_dir/dwmri_b0.nii.gz -i $in_dir/T1_5tt.nii.gz  -t [$in_dir/dwmri2T1_0GenericAffine.mat,1] -o $in_dir/dwmri_5tt.nii.gz  -n Linear
antsApplyTransforms -d 3 -e 0 -r $in_dir/dwmri_b0.nii.gz -i $in_dir/T1_gmwmi.nii.gz -t [$in_dir/dwmri2T1_0GenericAffine.mat,1] -o $in_dir/dwmri_gmwmi.nii.gz -n Linear

# Fit FOD
# - dwmri_tournier.txt
# - dwmri_fod.nii.gz

echo "prep_dwmri.sh: Fitting FOD..."
dwi2response tournier $in_dir/dwmri.nii.gz $in_dir/dwmri_tournier.txt -fslgrad $in_dir/dwmri.bvec $in_dir/dwmri.bval -mask $in_dir/dwmri_mask.nii.gz
dwi2fod csd $in_dir/dwmri.nii.gz $in_dir/dwmri_tournier.txt $in_dir/dwmri_fod.nii.gz -fslgrad $in_dir/dwmri.bvec $in_dir/dwmri.bval -mask $in_dir/dwmri_mask.nii.gz

# Move FOD to MNI space

echo "prep_dwmri.sh: Moving and reorienting FOD to MNI..."
ConvertTransformFile 3 $in_dir/dwmri2T1_0GenericAffine.mat $in_dir/dwmri2T1_0GenericAffine.txt
transformconvert $in_dir/dwmri2T1_0GenericAffine.txt itk_import $in_dir/dwmri2T1_0GenericAffine.trix
ConvertTransformFile 3 $in_dir/T12mni_0GenericAffine.mat $in_dir/T12mni_0GenericAffine.txt
transformconvert $in_dir/T12mni_0GenericAffine.txt itk_import $in_dir/T12mni_0GenericAffine.trix
mrtransform -linear $in_dir/dwmri2T1_0GenericAffine.trix -modulate fod -reorient_fod true $in_dir/dwmri_fod.nii.gz - | mrtransform -linear $in_dir/T12mni_0GenericAffine.trix -interp linear -template $in_dir/T1_N4_mni_2mm.nii.gz -stride $in_dir/T1_N4_mni_2mm.nii.gz -modulate fod -reorient_fod true - $in_dir/dwmri_fod_mni_2mm_trix.nii.gz

# Convert FOD to signal in MNI space

echo "prep_dwmri.sh: Sampling FOD in MNI space..."
scil_compute_sf_from_sh.py --sh_basis tournier07 --sphere repulsion100 $in_dir/dwmri_fod_mni_2mm_trix.nii.gz $in_dir/dwmri_sf_mni_2mm_trix.nii.gz

# Run tractography (.tck files will be deleted)
# - dwmri_train.tck (No ACT)
# - dwmri_test.tck  (With ACT)

# echo "prep_dwmri.sh: Generating training tractogram (without ACT)..."
# tckgen $in_dir/dwmri_fod.nii.gz $in_dir/dwmri_train.tck -algorithm SD_Stream -select 1000000 -step 1 -seed_image $in_dir/dwmri_seed.nii.gz -mask $in_dir/dwmri_mask.nii.gz -minlength 50
echo "prep_dwmri.sh: Generating testing tractogram (with ACT)..."
tckgen $in_dir/dwmri_fod.nii.gz $in_dir/dwmri_test50to250.tck  -algorithm SD_Stream -select 1000000 -step 1 -seed_image $in_dir/dwmri_seed.nii.gz -mask $in_dir/dwmri_mask.nii.gz -minlength 50 -maxlength 250 -act $in_dir/dwmri_5tt.nii.gz
# echo "prep_dwmri.sh: Generating testing tractogram (with ACT from GM/WM interface)..."
# tckgen $in_dir/dwmri_fod.nii.gz $in_dir/dwmri_gmwmi.tck -algorithm SD_Stream -select 1000000 -step 1 -seed_gmwmi $in_dir/dwmri_gmwmi.nii.gz -mask $in_dir/dwmri_mask.nii.gz -minlength 50 -act $in_dir/dwmri_5tt.nii.gz
# echo "prep_dwmri.sh: Generating probablistic tractogram (with ACT from GM/WM interface)..."
# tckgen $in_dir/dwmri_fod.nii.gz $in_dir/dwmri_prob.tck -select 1000000 -step 1 -seed_gmwmi $in_dir/dwmri_gmwmi.nii.gz -mask $in_dir/dwmri_mask.nii.gz -minlength 50 -act $in_dir/dwmri_5tt.nii.gz

# Move tractograms to T1 space (.trk files will be saved)
# - T1_train.trk (No ACT)
# - T1_test.trk  (With ACT)

echo "prep_dwmri.sh: Moving tractograms to T1 space..."
# scil_apply_transform_to_tractogram.py --reference $in_dir/dwmri_b0.nii.gz $in_dir/dwmri_train.tck $in_dir/T1_N4.nii.gz $in_dir/dwmri2T1_0GenericAffine.mat $in_dir/T1_train.trk --inverse --remove_invalid # --inverse needed per ANTs convention
scil_apply_transform_to_tractogram.py --reference $in_dir/dwmri_b0.nii.gz $in_dir/dwmri_test50to250.tck  $in_dir/T1_N4.nii.gz $in_dir/dwmri2T1_0GenericAffine.mat $in_dir/T1_test50to250.trk  --inverse --remove_invalid 
# scil_apply_transform_to_tractogram.py --reference $in_dir/dwmri_b0.nii.gz $in_dir/dwmri_gmwmi.tck $in_dir/T1_N4.nii.gz $in_dir/dwmri2T1_0GenericAffine.mat $in_dir/T1_gmwmi.trk --inverse --remove_invalid 
# scil_apply_transform_to_tractogram.py --reference $in_dir/dwmri_b0.nii.gz $in_dir/dwmri_prob.tck $in_dir/T1_N4.nii.gz $in_dir/dwmri2T1_0GenericAffine.mat $in_dir/T1_prob.trk --inverse --remove_invalid 

# Move from T1 to MNI space

echo "prep_dwmri.sh: Moving tractograms to MNI space..."
# scil_apply_transform_to_tractogram.py $in_dir/T1_train.trk $in_dir/T1_N4_mni_2mm.nii.gz $in_dir/T12mni_0GenericAffine.mat $in_dir/T1_train_mni_2mm.trk --inverse --remove_invalid # --inverse needed per ANTs convention
scil_apply_transform_to_tractogram.py $in_dir/T1_test50to250.trk  $in_dir/T1_N4_mni_2mm.nii.gz $in_dir/T12mni_0GenericAffine.mat $in_dir/T1_test50to250_mni_2mm.trk  --inverse --remove_invalid 
# scil_apply_transform_to_tractogram.py $in_dir/T1_gmwmi.trk $in_dir/T1_N4_mni_2mm.nii.gz $in_dir/T12mni_0GenericAffine.mat $in_dir/T1_gmwmi_mni_2mm.trk --inverse --remove_invalid 
# scil_apply_transform_to_tractogram.py $in_dir/T1_prob.trk $in_dir/T1_N4_mni_2mm.nii.gz $in_dir/T12mni_0GenericAffine.mat $in_dir/T1_prob_mni_2mm.trk --inverse --remove_invalid 

# Compute TDI in MNI space
echo "prep_dwmri.sh: Computing track density images..."
# scil_compute_streamlines_density_map.py $in_dir/T1_train_mni_2mm.trk $in_dir/T1_train_mni_2mm_tdi.nii.gz
scil_compute_streamlines_density_map.py $in_dir/T1_test50to250_mni_2mm.trk $in_dir/T1_test50to250_mni_2mm_tdi.nii.gz
# scil_compute_streamlines_density_map.py $in_dir/T1_gmwmi_mni_2mm.trk $in_dir/T1_gmwmi_mni_2mm_tdi.nii.gz
# scil_compute_streamlines_density_map.py $in_dir/T1_prob_mni_2mm.trk $in_dir/T1_prob_mni_2mm_tdi.nii.gz

# Wrap up

echo "prep_dwmri.sh: Done!"
