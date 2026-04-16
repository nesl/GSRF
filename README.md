

---

## 1. Environment Setup

```bash
/usr/bin/python3.10 -m venv .gsrf
source .gsrf/bin/activate

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -e submodules/simple-knn -e submodules/complex-gaussian-tracer -e submodules/complex-gaussian-tracer-csi

pip install tqdm plyfile matplotlib scikit-image lpips seaborn pyyaml
pip install "numpy<2"
```

## 2. Dataset and Pretrained Weights

Three datasets are used in this project:

| Dataset | Signal Type |
|---------|------------|
| RFID | Spectrum |
| BLE | Received Signal Strength Indicator (RSSI) |
| CSI | Channel State Information (CSI) |

All datasets are available at: https://github.com/XPengZhao/NeRF2

Place each dataset under `./data/`:

```
./data/rfid/
./data/ble_rssi/
./data/csi/
```

Pretrained weights can be downloaded below. Place them under `./weights/`.

| Dataset | Weights |
|---------|---------|
| RFID | [download](https://drive.google.com/file/d/1nA_dS0RElolKg4mleupt0MxeAhRE1DcT/view?usp=sharing) |
| BLE | [download](https://drive.google.com/file/d/1CUvPbWZn3Jqr6SxPznn7GqvrJH6w-Gtg/view?usp=sharing) |
| CSI | [download](https://drive.google.com/file/d/1w-FCEjzvbkSeFeXxdQz5vsuC9-PIDw_I/view?usp=sharing) |

```
./weights/rfid/
./weights/ble_rssi/
./weights/csi/
```

## 3. RFID

### 3.1 Training

```bash
bash run_rfid.sh
```

This runs both training and inference using `arguments/configs/rfid/exp1.yaml` on GPU 0 by default.

```bash
bash run_rfid.sh --config path/to.yaml  # use a different config
bash run_rfid.sh --train                # training only
bash run_rfid.sh --infer                # inference only
```

### 3.2 Inference from Pretrained Model

Pretrained weights (trained with [`rfid/exp1.yaml`](arguments/configs/rfid/exp1.yaml) settings) are available in [Section 2](#2-dataset-and-pretrained-weights).

```bash
python inference_rfid.py --config <config> \
    --model_path <model_dir> --output_dir <output_dir>

# example: using pretrained weights
python inference_rfid.py --config arguments/configs/rfid/exp1.yaml \
    --model_path weights/rfid --output_dir results/rfid
```

## 4. BLE RSSI

### 4.1 Training

```bash
bash run_ble.sh
```

This trains one model per gateway using `arguments/configs/ble/exp1.yaml` on GPU 0 by default. The first gateway is trained fully (geometry + FLE coefficients). Since all gateways share the same physical environment, subsequent gateways reuse the learned geometry and only train their FLE coefficients. Evaluation runs on the test set at each checkpoint.

```bash
bash run_ble.sh --config path/to.yaml  # use a different config
```

### 4.2 Inference from Pretrained Model

Pretrained weights for all 21 gateways (trained with [`ble/exp1.yaml`](arguments/configs/ble/exp1.yaml) settings) are available in [Section 2](#2-dataset-and-pretrained-weights).

```bash
python inference_ble.py --config <config> \
    --model_path <model_dir> --output_dir <output_dir>

# example: using pretrained weights
python inference_ble.py --config arguments/configs/ble/exp1.yaml \
    --model_path weights/ble_rssi --output_dir results/ble_rssi
```

## 5. CSI

### 5.1 Training

```bash
bash run_csi.sh
```

This runs two-phase training using `arguments/configs/csi/exp1.yaml` on GPU 0 by default:
- **Phase 1**: Autoencoder pretraining — learns an encoder that maps uplink CSI to 3D TX positions. The pretrained encoder is saved to `logs/csi/pretrained_encoder.pth` and reused on subsequent runs.
- **Phase 2**: Per-antenna Gaussian splatting — the first antenna is trained fully (geometry + FLE coefficients). Since all antennas share the same physical environment, subsequent antennas reuse the learned geometry and only train their FLE coefficients. Evaluation runs at each test checkpoint.

```bash
bash run_csi.sh --config path/to.yaml  # use a different config

# skip Phase 1 by providing a pretrained encoder
python main_csi.py --config arguments/configs/csi/exp1.yaml \
    --pretrained_encoder /path/to/pretrained_encoder.pth
```

### 5.2 Inference from Pretrained Model

Pretrained weights for all 8 antennas (trained with [`csi/exp1.yaml`](arguments/configs/csi/exp1.yaml) settings) are available in [Section 2](#2-dataset-and-pretrained-weights).

```bash
python inference_csi.py --config <config> \
    --model_path <model_dir> --output_dir <output_dir>

# example: using pretrained weights
python inference_csi.py --config arguments/configs/csi/exp1.yaml \
    --model_path weights/csi --output_dir results/csi
```

## 6. Acknowledgments

This codebase is adapted from [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) by the GraphDECO research group at Inria.

## 7. License

BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
