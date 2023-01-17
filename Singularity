Bootstrap: docker
From: ubuntu:18.04

%post -c /bin/bash

    cd /

    # Prepare directories for installing applications
    mkdir -p apps
    mkdir -p installers

    # Update all libraries
    apt-get -y update

    # Install xvfb
    apt-get -y install xvfb

    # Install ghostscript for pdf management
    apt-get -y install ghostscript

    # Install MRTrix3
    apt-get -y install git g++ python python-numpy libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev python3-distutils
    cd /apps
    git clone https://github.com/MRtrix3/mrtrix3.git
    cd mrtrix3
    git checkout 3.0.3
    ./configure
    ./build
    cd /

    # Install FSL
    apt-get -y install python wget ca-certificates libglu1-mesa libgl1-mesa-glx libsm6 libice6 libxt6 libpng16-16 libxrender1 libxcursor1 libxinerama1 libfreetype6 libxft2 libxrandr2 libgtk2.0-0 libpulse0 libasound2 libcaca0 libopenblas-base bzip2 dc bc 
    wget -O /installers/fslinstaller.py "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py"
    cd /installers
    python fslinstaller.py -d /apps/fsl -V 6.0.6
    cd /

    # Install ANTs (and compatible CMake)
    apt-get -y install build-essential libssl-dev
    # CMake: The latest ANTs requires newer version of cmake than can be installed
    # through apt-get, so we need to build higher version of cmake from source
    cd /installers
    mkdir cmake_install
    cd cmake_install
    wget https://github.com/Kitware/CMake/releases/download/v3.23.0-rc2/cmake-3.23.0-rc2.tar.gz
    tar -xf cmake-3.23.0-rc2.tar.gz
    cd cmake-3.23.0-rc2/
    ./bootstrap
    make
    make install
    cd /
    # ANTS
    cd /installers
    mkdir ants_installer
    cd ants_installer
    git clone https://github.com/ANTsX/ANTs.git
    git checkout efa80e3f582d78733724c29847b18f3311a66b54
    mkdir ants_build
    cd ants_build
    cmake /installers/ants_installer/ANTs -DCMAKE_INSTALL_PREFIX=/apps/ants
    make 2>&1 | tee build.log
    cd ANTS-build
    make install 2>&1 | tee install.log
    cd /
    
    # SCILPY
    apt-get -y install git gcc libpq-dev python-dev python-pip python3.8 python3.8-dev python3-pip python3.8-venv python3-wheel
    apt-get -y install libblas-dev liblapack-dev libfreetype6-dev pkg-config 
    apt-get -y install libglu1-mesa libgl1-mesa-glx libxrender1
    cd /apps
    git clone https://github.com/scilus/scilpy.git
    cd scilpy
    git checkout 1.4.1
    python3.8 --version
    python3.8 -m venv venv
    source venv/bin/activate
    pip3 install wheel
    pip install --upgrade pip
    pip install --upgrade setuptools
    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
    pip install -e .
    deactivate
    cd /

    # Make custom folders
    mkdir -p data

    # Set Permissions
    chmod 755 /apps
    chmod 755 /data
    
    # Install source code
    cd /
    apt-get -y install git gcc libpq-dev python-dev python-pip python3.8 python3.8-dev python3-pip python3.8-venv python3-wheel
    git clone https://github.com/MASILab/cornn_tractography.git
    cd cornn_tractography
    git checkout v1.0.0
    python3.8 -m venv venv
    source venv/bin/activate
    python3 --version
    pip3 install wheel
    pip install --upgrade pip
    pip install --upgrade setuptools
    bash install/pip.sh
    deactivate
    cd /

    # Clean up
    rm -r /installers

%environment

    # MRTrix3
    export PATH="/apps/mrtrix3/bin:$PATH"

    # FSL
    FSLDIR=/apps/fsl
    . ${FSLDIR}/etc/fslconf/fsl.sh
    PATH=${FSLDIR}/bin:${PATH}
    export FSLDIR PATH

    # ANTs
    export ANTSPATH=/apps/ants/bin/
    export PATH=${ANTSPATH}:$PATH

    # CUDA
    export CPATH="/usr/local/cuda/include:$CPATH"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="/usr/local/cuda"

%runscript

    xvfb-run -a --server-num=$((65536+$$)) --server-args="-screen 0 1600x1280x24 -ac" bash /cornn_tractography/src/generate.sh "$@"
    