## 1.Environment Requirements

```sh

# The following environment has been tested and verified:
OS: Ubuntu 20.04 or higher 
Python: 3.11 (Recommended)
CUDA: 12.1 
PyTorch: 2.5.1

```

## 2.Installation

```sh
# Step 1: Create Virtual Environment
conda create -n sewer3d python=3.11 -y
conda activate sewer3d

# Step 2: Install PyTorch and Dependencies
# We recommend using PyTorch 2.5.1 with CUDA 12.1 support.
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 

# Install other requirements
pip install sharedarray tqdm open3d

# Step 3: Compile PointOps
# pointops requires the CUDA compiler (nvcc) version to strictly match the PyTorch CUDA version.
# If your system CUDA is not 12.1, please follow the "CUDA Version Conflict" section below first.

cd lib/pointops
python3 setup.py install
```

## 3.Data preparation

```sh

# Download the dataset from the following link.
https://www.kaggle.com/datasets/liminghao123/dut-sewer3d-semantic-segmentation-s3dss-dataset/code

# Dataset path
# reate a folder named "data" in the root directory and place the dataset in this folder.
# Example: My dataset path：
/userData/gpulij/sewer3d/data

```

## 4.Troubleshooting: CUDA Version Conflict

```sh
# If you encounter a version mismatch during pointops compilation (e.g., system CUDA is 11.8 but PyTorch is 12.1), execute the following within your conda environment:
# Step 1.Install CUDA Toolkit 12.1 in Conda:
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit

# Step 2.Set Environment Variables:
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# Step 3.Clean and Recompile:
rm -rf build/ dist/ *.egg-info 
python3 setup.py install
```

## 5.Training

```sh
# To start training, use the following command. You can specify the GPU ID using CUDA_VISIBLE_DEVICES.
# Before running, modify line 91 of the train_sewer3d.py file: `classifier = Model(c=feat_dim, k=num_class, experiment_type='III').to(device)`.
# Change experiment_type='III' to experiment_type='Ⅳ'
# For a detailed explanation, please refer to the end of the GraphAttention_multiscaleV1.py file.

# Example: Training on GPU 2
CUDA_VISIBLE_DEVICES=2 python train_sewer3d.py
```
