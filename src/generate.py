# Generate CoRNN Tractogram (Inference)
# Leon Cai
# MASI Lab

# *** TODO MRTrix WORKING DIRS SHOULD BE IN OURS

# Set Up

import os
import subprocess
import argparse as ap
import numpy as np
import nibabel as nib
from datetime import datetime
from tqdm import tqdm

import torch
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from utils import vox2trid, vox2trii, triinterp
from modules import DetRNN, DetCNNFake, DetConvProj

# Shared Variables

ATLAS_DIR = '/home-local/cornn_tractography/atlas'
MODEL_DIR = '/home-local/cornn_tractography/model'
SRC_DIR   = '/home-local/cornn_tractography/src'
SOURCE_SCILPY = 'source ~/Apps/scilpy/venv/bin/activate'
SOURCE_VENV   = 'source /home-local/cornn_tractography/venv/bin/activate'

# Helper Functions

def tri2act(curr_trid, curr_trii, curr_step, prev_trid, prev_trii, prev_step, step, max_steps, act_img, mask_img, tissue_thres=0.5, angle=60, ignore_angle=False):
    
    # Compute curvature

    if prev_step is not None and not ignore_angle:
        cos_angle = np.cos(angle*np.pi/180)
        cos_vects = np.clip(np.sum(curr_step * prev_step, axis=1), -1, 1)
        high_curv = cos_vects < cos_angle
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
    exits_mask = curr_tri[:, OUT] > tissue_thres
    other_wm   = np.logical_and(np.logical_and(prev_tri[:, WM] > tissue_thres, curr_tri[:, WM] > tissue_thres), high_curv)
    other_dgm  = np.logical_and(np.logical_and(prev_tri[:, DGM] > tissue_thres, curr_tri[:, DGM] > tissue_thres), high_curv)
    exits_dgm  = np.logical_and(prev_tri[:, DGM] > tissue_thres, curr_tri[:, DGM] < tissue_thres)

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
            batch_rev_step, _, _, batch_rev_hidden, _ = rnn(ten, batch_rev_trid.to(device), batch_rev_trii)                         # Forward pass
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

def run(cmd):

    print(cmd)
    subprocess.check_call(cmd, shell=True, executable='/bin/bash')

# Go!

if __name__ == '__main__':

    # --------------
    # Prepare inputs
    # --------------

    parser = ap.ArgumentParser(description='CoRNN tractography: Streamline propagation with convolutional-recurrent neural networks')
    
    parser.add_argument('t1_file', metavar='/in/file.nii.gz', help='path to the input NIFTI file')
    parser.add_argument('trk_file', metavar='/out/file.trk', help='path to the output tractogram file')

    parser.add_argument('--slant', metavar='/slant/dir', default=None, help='path to the SLANT output directory (required)')
    parser.add_argument('--wml', metavar='/wml/dir', default=None, help='path to the WML TractSeg output directory (required)')

    parser.add_argument('--device', metavar='cuda/cpu', default='cpu', help='string indicating device on which to perform tracking (default = "cpu")')
    parser.add_argument('--num_streamlines', metavar='N', default='1000000', help='number of streamlines (default = 1000000)')
    parser.add_argument('--num_seeds', metavar='N', default='100000', help='number of streamline seeds per batch (default = 100000)')
    parser.add_argument('--min_steps', metavar='N', default='50', help='minimum number of 1mm steps for streamlines (default = 50)')
    parser.add_argument('--max_steps', metavar='N', default='250', help='maximum number of 1mm steps for streamlines (default = 250)')
    parser.add_argument('--buffer_steps', metavar='N', default='5', help='number of 1mm steps where the angle criteria is ignored at the beginning of tracking (default = 5)')
    parser.add_argument('--unidirectional', action='store_true', help='perform only unidirectional tracking (default = bidirectional)')

    parser.add_argument('--work_dir', metavar='/work/dir', default='/tmp/cornn_{}'.format(datetime.now().strftime('%m%d%Y_%H%M%S')), help='path to temporary working directory (default = make new directory in /tmp)')
    parser.add_argument('--keep_work', action='store_true', help='do NOT remove working directory')
    parser.add_argument('--num_threads', metavar='N', default=1, help='Non-negative integer indicating number of threads to use when running multi-threaded steps of this pipeline (default = 1)')

    args = parser.parse_args()

    # ------------
    # Parse inputs
    # ------------

    t1_file = args.t1_file
    assert os.path.exists(args.t1_file), 'Input T1 file {} does not exist. Aborting.'.format(t1_file)

    trk_file = args.trk_file
    trk_dir = os.path.dirname(trk_file)
    assert os.path.exists(trk_dir), 'Output directory {} does not exist. Aborting.'.format(trk_dir)
    
    slant_dir = str(args.slant)
    assert os.path.exists(slant_dir), 'SLANT directory {} does not exist. Aborting.'.format(slant_dir)
    wml_dir = str(args.wml)
    assert os.path.exists(wml_dir), 'WML TractSeg directory {} does not exist. Aborting.'.format(wml_dir)
        
    device_str = args.device
    
    num_seeds = int(args.num_seeds)
    assert num_seeds > 0, 'Parameter num_seeds must be positive. {} provided. Aborting.'.format(num_seeds)

    max_steps = int(args.max_steps)
    assert max_steps > 0, 'Parameter max_steps must be positive. {} provided. Aborting.'.format(max_steps)
    min_steps = int(args.min_steps)
    assert min_steps > 0, 'Parameter min_steps must be positive. {} provided. Aborting.'.format(min_steps)
    assert min_steps < max_steps, 'Parameter min_steps must be less than max_steps. {} and {} were provided.'.format(min_steps, max_steps)

    angle_steps = int(args.buffer_steps)
    assert angle_steps > 0, 'Parameter angle_steps must be positive. {} provided. Aborting.'.format(angle_steps)

    rev = not args.unidirectional

    num_select = int(args.num_streamlines)
    assert num_select > 0, 'Parameter num_streamlines must be positive. {} provided. Aborting.'.format(num_select)

    work_dir = args.work_dir
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    keep_work = args.keep_work

    # ---------------------------
    # Move into working directory
    # ---------------------------

    if not os.path.exists(os.path.join(work_dir, 'T1.nii.gz')):
        convert_cmd = 'mrconvert {} {}'.format(t1_file, os.path.join(work_dir, 'T1.nii.gz'))
        run(convert_cmd)

    # ----------
    # Prepare T1
    # ----------

    t1_cmd = '{} ; bash {} {} {} {} {} {}'.format(SOURCE_VENV, os.path.join(SRC_DIR, 'prep_T1.sh'), work_dir, ATLAS_DIR, SRC_DIR, slant_dir, wml_dir)
    run(t1_cmd)

    # -----------------
    # Prepare inference
    # -----------------

    # Load data

    mask_img    = nib.load(os.path.join(work_dir, 'T1_mask_mni_2mm.nii.gz')).get_fdata().astype(bool)
    seed_img    = nib.load(os.path.join(work_dir, 'T1_seed_mni_2mm.nii.gz')).get_fdata()

    t1_nii      = nib.load(os.path.join(work_dir, 'T1_N4_mni_2mm.nii.gz'))
    t1_img      = t1_nii.get_fdata()
    t1_ten      = torch.FloatTensor(np.expand_dims(t1_img / np.median(t1_img[mask_img]), axis=(0, 1)))

    act_img     = nib.load(os.path.join(work_dir, 'T1_5tt_mni_2mm.nii.gz')).get_fdata()[:, :, :, :-1]
    act_ten     = torch.FloatTensor(np.expand_dims(np.transpose(act_img, axes=(3, 0, 1, 2)), axis=0))

    tseg_img    = nib.load(os.path.join(work_dir, 'T1_tractseg_mni_2mm.nii.gz')).get_fdata()
    tseg_ten    = torch.FloatTensor(np.expand_dims(np.transpose(tseg_img, axes=(3, 0, 1, 2)), axis=0))
    
    slant_img   = nib.load(os.path.join(work_dir, 'T1_slant_mni_2mm.nii.gz')).get_fdata()
    slant_ten   = torch.FloatTensor(np.expand_dims(np.transpose(slant_img, axes=(3, 0, 1, 2)), axis=0))
    
    ten         = torch.cat((t1_ten, act_ten, tseg_ten, slant_ten), dim=1)
    aff         = t1_nii.affine

    # Load model

    device   = torch.device(device_str)
    cnn      = DetConvProj(123, 512, kernel_size=7)
    rnn      = DetRNN(512, fc_width=512, fc_depth=4, rnn_width=512, rnn_depth=2)
    cnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 't1_cnn_e1073s1.pt'), map_location=torch.device('cpu')))
    rnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 't1_rnn_e1073s1.pt'), map_location=torch.device('cpu')))
    
    # -----------------------
    # Inference (image-level)
    # -----------------------

    if not os.path.exists(os.path.join(work_dir, 'inference_mni_2mm.nii.gz')):
        ten = ten2features(ten, cnn, device)
        img = np.transpose(np.squeeze(ten.cpu().numpy(), axis=0), axes=(1, 2, 3, 0))
        nii = nib.Nifti1Image(img, aff)
        nib.save(nii, os.path.join(work_dir, 'inference_mni_2mm.nii.gz'))
    else:
        nii = nib.load(os.path.join(work_dir, 'inference_mni_2mm.nii.gz'))
        img = nii.get_fdata()
        ten = torch.FloatTensor(np.expand_dims(np.transpose(img, axes=(3, 0, 1, 2)), axis=0))

    # --------------------
    # Inference (tracking)
    # --------------------

    if not os.path.exists(os.path.join(work_dir, 'inference_mni_2mm.trk')):
        
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
        streamlines_selected = streamlines_selected[:num_select]
        sft = StatefulTractogram(streamlines_selected, reference=t1_nii, space=Space.VOX)
        save_tractogram(sft, os.path.join(work_dir, 'inference_mni_2mm.trk'), bbox_valid_check=False)

    # -------------------------------------------------
    # Post-processing and move out of working directory
    # -------------------------------------------------
    
    if not os.path.exists(trk_file):
        trk_cmd = '{} ; scil_apply_transform_to_tractogram.py {} {} {} {} --remove_invalid'.format(SOURCE_SCILPY,
                                                                                                   os.path.join(work_dir, 'inference_mni_2mm.trk'), 
                                                                                                   os.path.join(work_dir, 'T1_N4.nii.gz'), 
                                                                                                   os.path.join(work_dir, 'T12mni_0GenericAffine.mat'), 
                                                                                                   trk_file) # no --inverse needed per ANTs convention
        run(trk_cmd)
