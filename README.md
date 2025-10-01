
# GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis 

## 📑 Abstract

Synthesizing radio-frequency (RF) data given the transmitter and receiver positions (e.g., received signal strength indicator, RSSI) is critical for wireless networking and sensing applications, such as indoor localization.  
However, it remains challenging due to complex propagation interactions, including reflection, diffraction, and scattering.
State-of-the-art neural radiance field (NeRF)-based methods achieve high-fidelity RF data synthesis but are limited by long training times and high inference latency.  

We introduce **GSRF**, a framework that extends 3D Gaussian Splatting (3DGS) from the optical domain to the RF domain, enabling efficient RF data synthesis.  

**Key innovations:**
1. Complex-valued 3D Gaussians with a hybrid Fourier–Legendre basis to model directional and phase-dependent radiance.  
2. Orthographic splatting for efficient ray–Gaussian intersection identification.  
3. A complex-valued ray tracing algorithm, executed on RF-customized CUDA kernels and grounded in wavefront propagation principles, to synthesize RF data in real time.  

Evaluated across various RF technologies, **GSRF** preserves high-fidelity RF data synthesis while achieving significant improvements in training efficiency, shorter training time, and reduced inference latency.  

---

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
