#!/bin/bash
set -e

GPU="1"
CONFIG="arguments/configs/rfid/exp1.yaml"
MODE="both"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --gpu)    GPU="$2"; shift 2 ;;
        --train)  MODE="train"; shift ;;
        --infer)  MODE="infer"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ! -f "$CONFIG" ]; then echo "Error: Config not found: $CONFIG"; exit 1; fi

export CUDA_VISIBLE_DEVICES="$GPU"

EXP_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['exp_name'])")
DATASET=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['dataset'])")
EXP_DIR="logs/${DATASET}/${EXP_NAME}"

if [ "$MODE" = "both" ] || [ "$MODE" = "train" ]; then
    PYTHONUNBUFFERED=1 python train_rfid.py --config "$CONFIG"
fi

if [ ! -d "$EXP_DIR" ]; then echo "Error: Experiment directory not found: $EXP_DIR"; exit 1; fi

if [ "$MODE" = "both" ] || [ "$MODE" = "infer" ]; then
    ITERS=$(ls "$EXP_DIR"/chkpnt*.pth 2>/dev/null | grep -oP '\d+(?=\.pth)' | sort -n)
    if [ -z "$ITERS" ]; then echo "No checkpoints found in $EXP_DIR"; exit 1; fi

    for ITER in $ITERS; do
        PYTHONUNBUFFERED=1 python inference_rfid.py --config "$CONFIG" --iterations "$ITER"
    done
fi
