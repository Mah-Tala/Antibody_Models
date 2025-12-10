#!/usr/bin/env bash
set -euo pipefail

# --- Miniconda bootstrap (skip if you already have conda) ---
mkdir -p ~/miniconda3
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda activate base

# --- Create env ---
conda create -y -n protein python=3.10
conda activate protein

# --- Core deps ---
conda install -y -c anaconda git-lfs
# (Optional GPU runtimes; safe to keep; CPU works fine without)
conda install -y -c nvidia/label/cuda-12.4.0 cuda

conda install pandas

pip install torch               # CPU torch; replace with CUDA wheel if desired
pip install transformers==4.44.2
pip install accelerate==0.33.0
pip cache remove deepspeed
pip install deepspeed==0.14.4
pip install "huggingface_hub>=0.24"
pip install hf-transfer
pip install esm==3.1.4


echo "✅ Environment ready. Run: huggingface-cli login  (to access private models)"
