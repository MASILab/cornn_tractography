# CoRNN Tractography

![itscornn](https://github.com/MASILab/cornn_tractography/blob/master/CoRNN.png?raw=true)

Perform tractography on T1-weighted MRI, no diffusion MRI needed!

## Contents

* [Overview](#overview)
* [Authors and Reference](#authors-and-reference)
* [Containerization of Source Code](#containerization-of-source-code)
* [Command](#command)
* [Arguments and I/O](#arguments-and-io)
* [Options](#options)

## Overview

This repository contains the model weights, source code, and containerized implementation of convolutional-recurrent neural network (CoRNN) tractography on T1w MRI with associated [SLANT](https://github.com/MASILab/SLANTbrainSeg) and [WM learning (WML)](https://github.com/MASILab/WM_learning_release) TractSeg segmentations. 

## Authors and Reference

[Leon Y. Cai](mailto:leon.y.cai@vanderbilt.edu), Ho Hin Lee, Nancy R. Newlin, Cailey I. Kerley, Praitayini Kanakaraj, Qi Yang, Graham W. Johnson, Daniel Moyer, Kurt G. Schilling, Francois Rheault, and Bennett A. Landman. Convolutiona-recurrent neural networks approximate diffusion tractography from T1-weighted MRI and associated anatomical context. *In Submission*, 2023.

[Medical-image Analysis and Statistical Interpretation (MASI) Lab](https://my.vanderbilt.edu/masi), Vanderbilt University, Nashville, TN, USA

## Containerization of Source Code

    git clone https://github.com/MASILab/cornn_tractography.git
    cd /path/to/repo/cornn_tractography
    git checkout v1.0.0
    sudo singularity build /path/to/cornn_tractography.sif Singularity

We use Singularity version 3.8 CE with root permissions.

## Command

    singularity run 
    -e 
    --contain
    -B <t1_file>:/data/T1.nii.gz
    -B <out_file>:/data/tractogram.trk
    -B <slant_dir>:/data/slant
    -B <wml_dir>:/data/wml
    -B /tmp:/tmp
    -B /path/to/cuda:/usr/local/cuda
    --nv
    /path/to/cornn_tractography.sif
    /data/T1.nii.gz
    /data/tractogram.trk
    --slant /data/slant
    --wml /data/wml
    [options]
    
* Binding the `/tmp` directory is necessary when running the image with `--contain`.
* `--nv` and `-B /path/to/cuda:/usr/local/cuda` are optional. See options `--device`.

## Arguments and I/O

* **`<t1_file>`** Path on the host machine to the T1-weighted MRI with which tractography is to be performed in NIFTI format (either compressed or not).

* **`<out_file>`** Path on the host machine to the target tractogram to generate in trk, tck, vtk, fib, or dpy format.

* **`<slant_dir>`** Path on the host machine to the SLANT output directory

* **`<wml_dir>`** Path on the host machine to the TractSeg WM Learning output directory

## Options

**--device cuda/cpu**

A string indicating the device on which to perform inference.

Default = "cpu"

**--num_streamlines N**

A positive integer indicating the number of streamlines to identify.

Default = 1000000

**--num_seeds N**

A positive integer indicating the number of streamlines to seed per batch. One GB of GPU memory can handle approximately 10000 seeds.

Default = 100000

**--min_steps N**

A positive integer indicating the minimum number of 1mm steps per streamline.

Default = 50

**--max_steps N**

A positive integer indicating the maximum number of 1mm steps per streamline.

Default = 250

**--buffer_steps N**

A positive integer indicating the number of 1mm steps where the angle stopping criteria are ignored at the beginning of tracking.

Default = 5

**--unidirectional**

A flag indicating that bidirectional tracking should not be performed. The buffer steps are NOT removed in this case.

Default = Perform bidirectional tracking

**--work_dir /data/work_dir**

A string indicating the working directory to use. The location of the working directory on the host machine, `<work_dir>`, must also be bound into the container with `-B <work_dir>:/data/work_dir` in the [command](#command). If the working directory contains previously generated intermediates, the corresponding steps will not be rerun.

**--keep_work**

A flag indicating that the intermediates in the working directory should NOT be cleared.

Default = Clear working directory after completion

**--num_threads N**

A positive integer indicating the number of threads to use during multithreaded steps.

Default = 1

**--force**

A flag indicating that the output file should be overwritten if it already exists.

Default = Do NOT override existing output file
