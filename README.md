
# GSRF


This repository is tested on **NVIDIA H100** GPUs and is built upon the official [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) framework.

## 🛠️ Environment Setup

```bash
/usr/bin/python3.10 -m venv .gsrf
source .gsrf/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -e ./submodules/simple-knn -e ./submodules/complex-gaussian-tracer

pip install tqdm plyfile matplotlib scikit-image lpips seaborn pyyaml
pip install "numpy<2"
```

## 🧪 Training

```bash
python train.py
```

## 🔍 Inference

```bash
python inference.py
```

## 📁 Dataset

The RFID spectrum dataset is available at:

https://github.com/XPengZhao/NeRF2

Place the dataset under the following directory:

```bash
./data/
```

## 📌 Acknowledgments

This codebase is adapted from [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) by the GraphDECO research group at Inria.
