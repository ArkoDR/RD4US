# RD4US
This repository contains the official implementation of the paper "RD4US: Unsupervised Anomaly Detection for Deep Vein Thrombosis via Cross-Frame Reverse Distillation".

## Quick Start

### 1. Data Preparation
Prepare your ultrasound dataset in the following directory structure:
```
dataset/
├── train/
│ ├── patient_001/
│ │ ├── *_s.png
│ │ ├── *_k.png
│ │ └── *_e.png
│ ├── patient_002/
│ │ ├── *_s.png
│ │ ├── *_k.png
│ │ └── *_e.png
│ └── ...
└── test/
  ├── patient_003/
  │ ├── *_s.png
  │ ├── *_k.png
  │ └── *_e.png
  └── ...
```

### 2. Training
Train the model using the following command:
```bash
python train.py \
    --input_dir /path/to/your/dataset \
    --save_dir /path/to/save/models \
    --epoch 100 \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --size 256
```

### 3. Inference

Run anomaly detection on test data:
```bash
python evaluate.py \
    --model_path /path/to/saved/model.pth \
    --test_dir /path/to/test/data \
    --size 256
```

## Acknowledgements

This implementation builds upon and extends the [RD4AD](https://github.com/hq-deng/RD4AD) codebase. We thank the authors for their open-source contribution.
