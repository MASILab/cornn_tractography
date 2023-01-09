# Generate Deterministic Tractogram (Inference)
# Leon Cai
# MASI Lab
# August 15, 2022

# *** TODO ***
# *** improve tri2act <- biggest to do
# *** - step 1 is to use 5tt in 1mm space (need to scale voxel coordinates to 1mm space (x2))
# *** - dgm / all ACT criteria [done]
# *** include last step for terminating streamlines and change to 0.5 tissue thres default and 60deg angle default [done]
# *** maybe do these last when everything like tri2act is finalized:
# *** - shave off first angle_steps? max_steps param would be max_steps+angle_steps and then streamlines_vox[angle_steps+1:, :, :] with padding to match reverse?
# *** - streamlines_terminate may not be necessary (stop is encoded in nans)?? same with streamlines_reject, if we just remove them before returning...?
# *** - angle_steps could also be removed before returning as well, thus also removing future empty streamlines? 
# *** - improve arguments

# Set Up

import torch
from tqdm import tqdm
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import numpy as np
import nibabel as nib
import sys
import os
import argparse as ap

from utils import vox2trid, vox2trii, triinterp, axis2step, onehot, vox2step
from modules import DetCNN, DetRNN, DetCNNPooled, DetCNNField, DetCNNK, DetCNNX, DetCNNFake, DetCNNFOD, DetCNNFODCompress, DetCNNT1Compress, DetConvProj

# Helper Functions

def tri2act(curr_trid, curr_trii, curr_step, prev_trid, prev_trii, prev_step, step, max_steps, act_img, mask_img, tissue_thres=0.5, angle=60, ignore_angle=False):

    # ACT Pseudocode (https://www.sciencedirect.com/science/article/pii/S1053811912005824)
    # 1) Enters CGM -> Terminate
    # 2) Enters CSF -> Reject
    # 3) Exits mask -> Terminate
    # 4) Meets other termination criteria in WM -> Reject
    # 5) Meets other termination criteria in DGM -> Terminate
    # 6) Exits DGM (previously in WM) -> Terminate 
    #
    # Examples of "other termination criteria":
    # - Low FA
    # - Low FOD amplitude
    # - High curvature
    #
    # Notes for us:
    # - High curvature is the only relevant "other termination criteria"
    # - When seeding from WM, (6) is equivalent to any exit of DGM
    
    # Compute curvature

    if prev_step is not None and not ignore_angle:
        cos_angle = np.cos(angle*np.pi/180)
        cos_vects = np.clip(np.sum(curr_step * prev_step, axis=1), -1, 1)
        high_curv = cos_vects < cos_angle # angle greater than threshold => cos less than cos of threshold
    else:
        high_curv = np.zeros((curr_step.shape[0])).astype(bool)

    # Compute tissue classes

    img = np.concatenate((act_img, np.expand_dims(np.logical_not(mask_img), axis=3)), axis=3)

    CGM, DGM, WM, CSF, OUT = 0, 1, 2, 3, 4

    logical_any = lambda *x : np.any(np.stack(x, axis=1), axis=1)
    logical_all = lambda *x : np.all(np.stack(x, axis=1), axis=1) 

    curr_tri = triinterp(img, curr_trid, curr_trii)
    prev_tri = triinterp(img, prev_trid, prev_trii)

    enters_cgm = np.logical_and(prev_tri[:, CGM] < curr_tri[:, CGM], curr_tri[:, CGM] > tissue_thres)
    enters_csf = np.logical_and(prev_tri[:, CSF] < curr_tri[:, CSF], curr_tri[:, CSF] > tissue_thres)
    exits_mask = curr_tri[:, OUT] > tissue_thres # np.logical_and(prev_tri[:, OUT] < curr_tri[:, OUT], curr_tri[:, OUT] > tissue_thres)
    other_wm   = np.logical_and(np.logical_and(prev_tri[:, WM] > tissue_thres, curr_tri[:, WM] > tissue_thres), high_curv)
    other_dgm  = np.logical_and(np.logical_and(prev_tri[:, DGM] > tissue_thres, curr_tri[:, DGM] > tissue_thres), high_curv)
    exits_dgm  = np.logical_and(prev_tri[:, DGM] > tissue_thres, curr_tri[:, DGM] < tissue_thres) # this is the tricky one.
    # other_dgm  = logical_all(
    #     prev_tri[:, DGM] > tissue_thres,
    #     prev_tri[:, WM] < tissue_thres,
    #     curr_tri[:, DGM] > tissue_thres,
    #     curr_tri[:, WM] < tissue_thres,
    #     high_curv
    # )
    # exits_dgm  = logical_all(
    #     prev_tri[:, DGM] > tissue_thres,
    #     prev_tri[:, WM] < tissue_thres,
    #     prev_tri[:, DGM] > curr_tri[:, DGM]
    # )

    reject = logical_any(enters_csf, other_wm)
    terminate = logical_any(enters_cgm, exits_mask, other_dgm, exits_dgm)
    terminate = np.logical_or(terminate, step >= max_steps-1)
    terminate = np.logical_and(terminate, np.logical_not(reject))

    return terminate, reject

def img2seeds(seed_img):

    seed_vox = np.argwhere(seed_img)
    seed_vox = seed_vox[np.random.choice(list(range(len(seed_vox))), size=num_seeds, replace=True), :]
    seed_vox = seed_vox + np.random.rand(*seed_vox.shape) - 0.5

    return seed_vox

def seeds2streamlines(seed_vox, seed_step, seed_trid, seed_trii, seed_hidden, seed_max_steps, angle_steps, rnn, ten, device):

    # Prep outputs

    num_seeds = seed_vox.shape[0]
    streamlines_vox = [seed_vox]                                # Save current floating points in voxel space
    streamlines_terminate = np.zeros((num_seeds,)).astype(bool) # Save cumulative ACT status
    streamlines_reject = np.zeros((num_seeds,)).astype(bool)
    streamlines_active = np.logical_not(np.logical_or(streamlines_terminate, streamlines_reject))

    # Prep inputs

    prev_vox, prev_step, prev_trid, prev_trii, prev_hidden = seed_vox, seed_step, seed_trid, seed_trii, seed_hidden

    # Track
    
    rnn = rnn.to(device)
    rnn.eval()
    step_bar = tqdm(range(np.max(seed_max_steps)))
    step_bar.set_description('Batch {}'.format(batch))
    for step in step_bar:
    
        # Given information about previous point, compute/analyze current point

        with torch.no_grad():
            curr_step, _, _, curr_hidden, _ = rnn(ten.to(device), torch.FloatTensor(prev_trid).to(device), torch.LongTensor(prev_trii), h=prev_hidden.to(device))
        
        curr_step = curr_step[0].cpu().numpy()
        curr_vox = prev_vox + curr_step / 2 # 2mm resolution to 1mm steps (1mm steps in 2mm voxel space have length 0.5)
        curr_trid = vox2trid(curr_vox)
        curr_trii = vox2trii(curr_vox, t1_img)
        
        curr_terminate, curr_reject = tri2act(curr_trid, curr_trii, curr_step, prev_trid, prev_trii, prev_step, step, seed_max_steps, act_img, mask_img, ignore_angle=step<angle_steps)

        # Update streamlines criteria: ACT

        streamlines_subset = np.logical_not(np.logical_or(streamlines_terminate, streamlines_reject)) # subset currently being considered among all seeds

        streamlines_terminate[streamlines_active] = np.logical_or(streamlines_terminate[streamlines_active], curr_terminate) # terminated/rejected/active among all seeds
        streamlines_reject[streamlines_active] = np.logical_or(streamlines_reject[streamlines_active], curr_reject)
        streamlines_active = np.logical_not(np.logical_or(streamlines_terminate, streamlines_reject))

        streamlines_subset_active = streamlines_active[streamlines_subset] # active among subset

        # Mark terminated or rejected streamlines

        curr_vox_nan = np.empty(seed_vox.shape)
        curr_vox_nan[:] = np.nan
        # curr_vox_nan[np.logical_and(streamlines_subset, streamlines_active), :] = curr_vox[streamlines_subset_active]
        curr_vox_nan[np.logical_and(streamlines_subset, np.logical_not(streamlines_reject))] = curr_vox[np.logical_not(streamlines_reject)[streamlines_subset]]
        streamlines_vox.append(curr_vox_nan)

        # Update Loop

        streamlines_active_num = num_seeds - np.sum(np.logical_not(streamlines_active))
        step_bar.set_description('Batch {}: {} seeds, {} ({:0.2f}%) terminated, {} ({:0.2f}%) rejected, {} ({:0.2f}%) active'.format(batch, num_seeds, np.sum(streamlines_terminate), 100*np.sum(streamlines_terminate)/num_seeds, np.sum(streamlines_reject), 100*np.sum(streamlines_reject)/num_seeds, streamlines_active_num, 100*streamlines_active_num/num_seeds))
        if streamlines_active_num < 2: # batch norm
            break

        prev_vox, prev_step, prev_trid, prev_trii, prev_hidden = curr_vox[streamlines_subset_active, :], curr_step[streamlines_subset_active, :], curr_trid[streamlines_subset_active, :], curr_trii[streamlines_subset_active, :], curr_hidden[:, streamlines_subset_active, :]
        seed_max_steps = seed_max_steps[streamlines_subset_active]

    rnn = rnn.cpu()
    ten = ten.cpu()
    del prev_vox, prev_step, prev_trid, prev_trii, prev_hidden, curr_vox, curr_step, curr_trid, curr_trii, curr_hidden
    
    # Reformat streamlines

    streamlines_vox = np.stack(streamlines_vox, axis=2)
    streamlines_vox = np.transpose(streamlines_vox, axes=(2, 1, 0)) # seq x feature x batch

    return streamlines_vox, streamlines_terminate, streamlines_reject

def ten2features(ten, cnn, device):

    cnn = cnn.to(device)
    cnn.eval()
    with torch.no_grad():
        ten = cnn(ten.to(device))
    cnn = cnn.cpu()
    ten = ten.cpu()
    return ten.cpu()

def streamlines2reverse(streamlines_vox, streamlines_terminate, streamlines_reject, rnn, ten, device):

    # Reverse and reject streamlines and convert to trii/trid, noting NaNs

    rev_vox = np.flip(streamlines_vox, axis=0).copy()                                               # Flip streamlines
    
    rev_vox_nan = np.isnan(rev_vox)                                                                 # Identify NaNs and convert to -1's for trid/trii calculation
    rev_vox[rev_vox_nan] = -1

    rev_valid = np.logical_not(np.logical_or(streamlines_reject, np.all(rev_vox_nan, axis=(0, 1)))) # Remove rejected and empty (all NaN) streamlines
    rev_vox = rev_vox[:, :, rev_valid]
    rev_vox_nan = rev_vox_nan[:, :, rev_valid]
    rev_valid_num = np.sum(rev_valid)

    rev_trid = vox2trid(rev_vox)                                                                    # Convert voxel locations to trid/trii for network
    rev_vox_packed = np.reshape(np.transpose(rev_vox, axes=(0, 2, 1)), (-1, 3))
    rev_trii_packed = vox2trii(rev_vox_packed, seed_img)
    rev_trii = np.transpose(np.reshape(rev_trii_packed, (rev_vox.shape[0], rev_vox.shape[2], -1)), axes=(0, 2, 1))

    # Remove NaNs from streamlines and convert to list of non-uniform length streamlines (tensors for PyTorch)

    rev_trid = np.split(rev_trid, indices_or_sections=rev_trid.shape[2], axis=2)                    # Split array into [seq x feat x streamlines=1] list
    rev_trii = np.split(rev_trii, indices_or_sections=rev_trii.shape[2], axis=2)
    rev_vox_nan = np.all(rev_vox_nan, axis=1)                                                       # Identify indices to remove along seq dimension for each streamline
    rev_lens = []
    for i in range(rev_valid_num):
        rev_trid[i] = torch.FloatTensor(rev_trid[i][np.logical_not(rev_vox_nan[:, i]), :, 0])       # Remove NaNs from seq dimension
        rev_trii[i] = torch.LongTensor(rev_trii[i][np.logical_not(rev_vox_nan[:, i]), :, 0])
        rev_lens.append(np.sum(np.logical_not(rev_vox_nan[:, i])))                                  # Record how long the remaining streamlines are for indexing

    # Compute hidden states batch-wise

    num_batches = np.median(np.sum(rev_vox_nan, axis=0)).astype(int)                                # Make each batch contain roughly the same number of total steps as there are streamlines
    rev_batch_idxs = np.round(np.linspace(0, rev_valid_num, num_batches+1)).astype(int)             # since we know we can fit 1 step per streamline and all streamlines on the GPU    

    rnn = rnn.to(device)
    rnn.eval()
    ten = ten.to(device)
    rev_step = []
    rev_hidden = []
    for i in tqdm(range(len(rev_batch_idxs)-1), desc='Batch {}: {} seeds, {} ({:0.2f}%) preparing for reverse tracking'.format(batch, num_seeds, rev_valid_num, 100*np.sum(rev_valid_num)/num_seeds)):
        batch_rev_trid = torch.nn.utils.rnn.pack_sequence(rev_trid[rev_batch_idxs[i]:rev_batch_idxs[i+1]], enforce_sorted=False)    # Pack sequences
        batch_rev_trii = torch.nn.utils.rnn.pack_sequence(rev_trii[rev_batch_idxs[i]:rev_batch_idxs[i+1]], enforce_sorted=False)
        batch_rev_lens = torch.LongTensor(rev_lens[rev_batch_idxs[i]:rev_batch_idxs[i+1]])                                          # Record length of each
        with torch.no_grad():
            batch_rev_step, _, _, batch_rev_hidden, _ = rnn(ten, batch_rev_trid.to(device), batch_rev_trii)                            # Forward pass
            batch_rev_step = batch_rev_step.cpu()
            batch_rev_hidden = batch_rev_hidden.cpu()
        batch_rev_step = torch.stack([batch_rev_step[batch_rev_lens[k]-1, k, :] for k in range(batch_rev_step.shape[1])], dim=0)    # Extract last step of each
        rev_step.append(batch_rev_step)
        rev_hidden.append(batch_rev_hidden)
    del batch_rev_step, batch_rev_hidden
    rnn = rnn.cpu()
    ten = ten.cpu()

    # Format for seeds2streamlines

    seed_step = np.concatenate(rev_step, axis=0)
    seed_vox = np.transpose(rev_vox[-1, :, :], axes=(1, 0)) + seed_step / 2 # (1mm steps in 2mm voxel space have length 0.5)
    seed_trid = vox2trid(seed_vox)
    seed_trii = vox2trii(seed_vox, t1_img)
    seed_hidden = torch.cat(rev_hidden, dim=1)
    seed_max_steps = max_steps - np.array(rev_lens)

    return seed_vox, seed_step, seed_trid, seed_trii, seed_hidden, seed_max_steps, rev_valid

# Go!

if __name__ == '__main__':

    # Parse inputs

    model_dir = '/nfs/masi/caily/projects/dt1/code/pilot/tensorboard/pilot_hcp_100_bestfodrnn_t1k7'
    suffix = '_e1073s1'
    fod_cnn_file = os.path.join(model_dir, 'fod_cnn{}.pt'.format(suffix))
    fod_rnn_file = os.path.join(model_dir, 'fod_rnn{}.pt'.format(suffix))
    t1_cnn_file = os.path.join(model_dir, 't1_cnn{}.pt'.format(suffix))
    t1_rnn_file = os.path.join(model_dir, 't1_rnn{}.pt'.format(suffix))

    in_dir = sys.argv[1]
    device_str = sys.argv[2]
    t1_file = os.path.join(in_dir, 'T1_N4_mni_2mm.nii.gz')
    mask_file = os.path.join(in_dir, 'T1_mask_mni_2mm.nii.gz')
    act_file = os.path.join(in_dir, 'T1_5tt_mni_2mm.nii.gz')
    # fs_file = os.path.join(in_dir, 'T1_fs_mni_2mm.nii.gz')
    seed_file = os.path.join(in_dir, 'T1_seed_mni_2mm.nii.gz')
    fod_file = os.path.join(in_dir, 'dwmri_fod_mni_2mm_trix.nii.gz')
    sf_file = os.path.join(in_dir, 'dwmri_sf_mni_2mm_trix.nii.gz')
    tseg_file = os.path.join(in_dir, 'T1_tractseg_mni_2mm.nii.gz')
    slant_file = os.path.join(in_dir, 'T1_slant_mni_2mm.nii.gz')
    posenc_file = os.path.join(in_dir, 'T1_posenc16_mni_2mm.nii.gz')
    out_file = os.path.join(in_dir, 'T1_gen_pilot_hcp_100_bestfodrnn_t1k7{}_mni_2mm_1mil_ACT.trk'.format(suffix))

    num_seeds = 180000
    max_steps = 250
    min_steps = 50
    angle_steps = 5
    rev = True
    num_select = 1000000
    
    # Load model

    img_shape = (96, 114, 96)
    device = torch.device(device_str)
    fod_cnn = DetCNNFake()
    t1_cnn  = DetConvProj(123, 512, kernel_size=7)
    fod_rnn = DetRNN(45, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2)
    t1_rnn  = DetRNN(512, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2)
    fod_cnn.load_state_dict(torch.load(fod_cnn_file, map_location=torch.device('cpu')))
    t1_cnn.load_state_dict(torch.load(t1_cnn_file, map_location=torch.device('cpu')))
    fod_rnn.load_state_dict(torch.load(fod_rnn_file, map_location=torch.device('cpu')))
    t1_rnn.load_state_dict(torch.load(t1_rnn_file, map_location=torch.device('cpu')))

    # Load images

    t1_nii = nib.load(t1_file)
    t1_img = t1_nii.get_fdata()

    seed_nii = nib.load(seed_file)
    seed_img = seed_nii.get_fdata()

    act_nii = nib.load(act_file)
    act_img = act_nii.get_fdata()[:, :, :, :-1]

    # fs_nii = nib.load(fs_file)
    # fs_img = fs_nii.get_fdata()

    mask_nii = nib.load(mask_file)
    mask_img = mask_nii.get_fdata().astype(bool)

    fod_nii = nib.load(fod_file)
    fod_img = fod_nii.get_fdata()

    sf_nii = nib.load(sf_file)
    sf_img = sf_nii.get_fdata()

    tseg_nii = nib.load(tseg_file)
    tseg_img = tseg_nii.get_fdata()

    slant_nii = nib.load(slant_file)
    slant_img = slant_nii.get_fdata()

    posenc_nii = nib.load(posenc_file)
    posenc_img = posenc_nii.get_fdata()

    t1_ten = torch.FloatTensor(np.expand_dims(t1_img / np.median(t1_img[mask_img]), axis=(0, 1)))
    act_ten = torch.FloatTensor(np.expand_dims(np.transpose(act_img, axes=(3, 0, 1, 2)), axis=0))
    fod_ten = torch.FloatTensor(np.expand_dims(np.transpose(fod_img, axes=(3, 0, 1, 2)), axis=0))
    sf_ten = torch.FloatTensor(np.expand_dims(np.transpose(sf_img, axes=(3, 0, 1, 2)), axis=0))
    tseg_ten = torch.FloatTensor(np.expand_dims(np.transpose(tseg_img, axes=(3, 0, 1, 2)), axis=0))
    slant_ten = torch.FloatTensor(np.expand_dims(np.transpose(slant_img, axes=(3, 0, 1, 2)), axis=0))
    posenc_ten = torch.FloatTensor(np.expand_dims(np.transpose(posenc_img, axes=(3, 0, 1, 2)), axis=0))
    
    # Embed and save imaging

    fod_pred_ten = ten2features(fod_ten, fod_cnn, device)
    fod_pred_file = out_file.replace('.trk', '_fod.nii.gz')
    # t1_pred_ten = ten2features(torch.cat((t1_ten, act_ten, tseg_ten), dim=1), t1_cnn, device)
    t1_pred_ten = ten2features(torch.cat((t1_ten, act_ten, tseg_ten, slant_ten), dim=1), t1_cnn, device)
    # t1_pred_ten = ten2features(torch.cat((t1_ten, act_ten, tseg_ten, slant_ten, posenc_ten), dim=1), t1_cnn, device)
    t1_pred_file = out_file.replace('.trk', '_t1.nii.gz')

    def save_cnn_pred_ten(pred_ten, pred_file, aff):
        pred_img = np.transpose(np.squeeze(pred_ten.cpu().numpy(), axis=0), axes=(1, 2, 3, 0))
        pred_nii = nib.Nifti1Image(pred_img, aff)
        nib.save(pred_nii, pred_file)
        print('Intermediate saved to {}'.format(pred_file))

    save_cnn_pred_ten(fod_pred_ten, fod_pred_file, fod_nii.affine)
    save_cnn_pred_ten(t1_pred_ten, t1_pred_file, t1_nii.affine)
    
    ten = t1_pred_ten
    rnn = t1_rnn

    # Loop through batches until number of desired streamlines are obtained

    streamlines_selected = []
    streamlines_selected_num = 0
    batch = 0

    while streamlines_selected_num < num_select:

        # Forward tracking

        # Generate forward seeds

        seed_vox = img2seeds(seed_img)                                  # Floating point location in voxel space
        seed_step = None                                                # Incoming angle to seed_vox (cartesian)
        seed_trid = vox2trid(seed_vox)                                  # Distance to lower voxel in voxel space
        seed_trii = vox2trii(seed_vox, seed_img)                        # Neighboring integer points in voxel space
        seed_hidden = torch.zeros((2, num_seeds, 512))                  # Hidden state (initialize as zeros per PyTorch docs)
        seed_max_steps = np.ones((num_seeds,)).astype(int) * max_steps  # Max number of propagating steps for each seed

        # Track

        streamlines_vox, streamlines_terminate, streamlines_reject = seeds2streamlines(seed_vox, seed_step, seed_trid, seed_trii, seed_hidden, seed_max_steps, angle_steps, rnn, ten, device)

        # Reverse tracking

        if rev:

            # Generate reverse seeds

            rev_vox, rev_step, rev_trid, rev_trii, rev_hidden, rev_max_steps, rev_valid = streamlines2reverse(streamlines_vox[angle_steps:, :, :], streamlines_terminate, streamlines_reject, rnn, ten, device)
            
            # Track

            rev_streamlines_vox, rev_streamlines_terminate, rev_streamlines_reject = seeds2streamlines(rev_vox, rev_step, rev_trid, rev_trii, rev_hidden, rev_max_steps, 0, rnn, ten, device)
            
            # Merge forward and reverse

            joint_streamlines_vox = np.empty((rev_streamlines_vox.shape[0], rev_streamlines_vox.shape[1], num_seeds))
            joint_streamlines_vox[:, :, rev_valid] = np.flip(rev_streamlines_vox, axis=0)
            joint_streamlines_vox[:, :, np.logical_not(rev_valid)] = np.nan
            joint_streamlines_vox = np.concatenate((joint_streamlines_vox, streamlines_vox[angle_steps:, :, :]), axis=0)

            streamlines_vox = joint_streamlines_vox
            streamlines_reject[rev_valid] = np.logical_or(streamlines_reject[rev_valid], rev_streamlines_reject)
        
        # Reformat streamlines and remove those that are too short

        streamlines_vox = np.split(streamlines_vox, streamlines_vox.shape[2], axis=2)
        streamlines_empty = np.zeros((len(streamlines_vox),)).astype(bool)
        for i in range(len(streamlines_vox)):
            streamlines_vox[i] = np.squeeze(streamlines_vox[i])
            keep_idxs = np.logical_not(np.all(np.isnan(streamlines_vox[i]), axis=1))
            num_steps = np.sum(keep_idxs)
            if num_steps == 0:
                streamlines_empty[i] = True
            if num_steps < min_steps + 1:
                streamlines_reject[i] = True
            streamlines_vox[i] = streamlines_vox[i][keep_idxs, :]
        streamlines_vox = nib.streamlines.array_sequence.ArraySequence(streamlines_vox)
        streamlines_vox = streamlines_vox[np.logical_not(streamlines_reject[np.logical_not(streamlines_empty)])]

        # Update selected

        streamlines_selected.append(streamlines_vox)
        streamlines_selected_num += len(streamlines_vox)
        print('Batch {}: {} seeds, {} ({:0.2f}%) selected, {} ({:0.2f}%) total'.format(batch, num_seeds, len(streamlines_vox), 100*len(streamlines_vox)/num_seeds, streamlines_selected_num, 100*streamlines_selected_num/num_select))

        # Prepare for next batch

        batch += 1

    # Save tractogram

    streamlines_selected = nib.streamlines.array_sequence.concatenate(streamlines_selected, axis=0)
    sft = StatefulTractogram(streamlines_selected, reference=t1_nii, space=Space.VOX)
    save_tractogram(sft, out_file, bbox_valid_check=False)
    print('Tractogram saved to {}'.format(out_file))
    