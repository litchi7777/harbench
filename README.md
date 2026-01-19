# HARBench

A comprehensive benchmark for evaluating foundation models in sensor-based Human Activity Recognition (HAR).

**Leaderboard**: https://litchi7777.github.io/harbench/

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.x (recommended)

## Quick Start (Docker)

```bash
# Build image
docker build -t harbench .

# Start container
docker run -dit --gpus all --name harbench harbench bash

# Enter container
docker exec -it harbench bash

# Inside container: preprocess and run benchmark
python preprocess.py --dataset dsads pamap2 mhealth realdisp mex forthtrace harth imwsha paal realworld selfback ucaehar uschad ward lara openpack exoskeletons vtt_coniot --download
python run_benchmark.py --model mtl --eval all
```

## Quick Start (Local)

```bash
# Create virtual environment
python -m venv .env
source .env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Preprocess all 18 fine-tuning datasets
python preprocess.py --dataset dsads pamap2 mhealth realdisp mex forthtrace harth imwsha paal realworld selfback ucaehar uschad ward lara openpack exoskeletons vtt_coniot --download

# Run benchmark
python run_benchmark.py --model mtl --eval all
```

## Project Structure

```
har-bench2/
├── finetune.py          # Fine-tuning and evaluation
├── pretrain.py          # Self-supervised pretraining
├── run_benchmark.py     # Full benchmark runner
├── src/
│   ├── data/            # Data loading utilities
│   └── models/          # Model definitions
├── pretrained/          # Pretrained weights
│   ├── mtl.pth          # (included)
│   ├── simclr.pth       # (included)
│   ├── moco.pth         # (included)
│   ├── cpc.pth          # (included)
│   ├── timemask.pth     # (included)
│   ├── timechannel.pth  # (included)
│   ├── limubert.pt.url  # (download separately)
│   ├── imumae.pth.url   # (download separately)
│   └── selfpab.ckpt.url # (download separately)
├── har-datasets/        # Dataset preprocessors
└── results/             # Experiment results
```

## Supported Models (13 total)

| Model | Type | Description | Weights |
|-------|------|-------------|---------|
| resnet | Baseline | 1D ResNet (random init) | - |
| mtl | SSL | Multi-Task Learning | Included |
| simclr | SSL | SimCLR | Included |
| moco | SSL | MoCo | Included |
| timechannel | SSL | Masked Resnet (time+channel) | Included |
| timemask | SSL | Masked Resnet (time only) | Included |
| cpc | SSL | Contrastive Predictive Coding | Included |
| harnet | Pretrained | OxWearables HARNet | Auto-download |
| selfpab | Transformer | SelfPAB (STFT + Transformer) | Download required |
| limubert | Transformer | LIMU-BERT | Download required |
| imumae | Transformer | IMU-Video-MAE (ECCV 2024) | Download required |
| patchtst | Foundation | PatchTST | Auto-download |
| moment | Foundation | MOMENT | Auto-download |

## Data Preparation

### Preprocessing Datasets

```bash
# List available datasets
python preprocess.py --list

# Preprocess a single dataset with auto-download
python preprocess.py --dataset dsads --download

# Preprocess all 18 fine-tuning datasets
python preprocess.py --dataset dsads pamap2 mhealth realdisp mex forthtrace harth imwsha paal realworld selfback ucaehar uschad ward lara openpack exoskeletons vtt_coniot --download

# Preprocess all 14 pretraining datasets (excluding NHANES)
python preprocess.py --dataset adlrd chad capture24 dog har70plus hhar imsb kddi_kitchen_left kddi_kitchen_right motionsense opportunity sbrhapt tmd wisdm --download

# NHANES is very large and takes a long time to process - run separately
python preprocess.py --dataset nhanes --download
```

Processed data will be saved to `har-datasets/data/processed/{dataset}/`

### Supported Datasets

**Fine-tuning Datasets (18)**

| Dataset | Sensors | Classes | Domain |
|---------|---------|---------|--------|
| DSADS | 5 | 19 | Exercise |
| PAMAP2 | 3 | 18 (12) | Daily |
| MHEALTH | 3 | 12 | Exercise |
| RealDisp | 9 | 33 | Exercise |
| MEX | 2 | 7 | Exercise |
| Forthtrace | 5 | 16 (11) | Daily |
| HARTH | 2 | 12 (10) | Daily |
| IMWSHA | 3 | 11 | Daily |
| PAAL | 1 | 24 (10) | Daily |
| RealWorld | 7 | 8 | Daily |
| SelfBack | 2 | 9 | Daily |
| UCAEHAR | 1 | 8 (6) | Daily |
| USC-HAD | 1 | 12 | Daily |
| WARD | 5 | 13 | Daily |
| LARa | 5 | 8 (6) | Industry |
| OpenPack | 4 | 10 (9) | Industry |
| Exoskeletons | 5 | 4 | Industry |
| VTT-ConIoT | 3 | 16 | Industry |

**Pretraining Datasets (14)**

| Dataset | Sensors | Description |
|---------|---------|-------------|
| NHANES | 1 | Large-scale (~13K subjects) |
| Capture-24 | 1 | Large-scale (151 subjects) |
| ADLRD | 1 | Daily activities |
| CHAD | 1 | Daily activities |
| Dog | 3 | Animal activity |
| HAR70+ | 2 | Elderly activities |
| HHAR | 7 | Device heterogeneity |
| IMSB | 2 | Daily activities |
| KDDI-Kitchen | 1 | Kitchen activities |
| MotionSense | 1 | Daily activities |
| Opportunity | 7 | Daily activities |
| SBR-HAPT | 1 | Daily activities |
| TMD | 1 | Transportation |
| WISDM | 2 | Daily activities |


## Usage

### Running Benchmark

Run comprehensive evaluation across multiple datasets:

```bash
# Run all evaluations for a model
python run_benchmark.py --model mtl --eval all

# Run specific evaluation types
python run_benchmark.py --model mtl --eval average    # Average performance
python run_benchmark.py --model mtl --eval domain     # Cross-domain evaluation
python run_benchmark.py --model mtl --eval position   # Sensor position evaluation
python run_benchmark.py --model mtl --eval fewshot    # Few-shot learning
python run_benchmark.py --model mtl --eval zeroshot   # Zero-shot evaluation

# Run on specific datasets
python run_benchmark.py --model mtl --datasets dsads pamap2 mhealth

# Multi-GPU parallel execution
python run_benchmark.py --model mtl --num_gpus 4 --parallel 2
```

Results will be saved to `results/benchmark/`.

### Fine-tuning (Single Run)

```bash
python finetune.py --model mtl --dataset dsads --sensors LeftArm --epochs 50
```

### Pretraining

Train your own SSL models:

```bash
python pretrain.py --method mtl --device cuda:0
```

## External Model Weights

Some models require downloading weights from external sources:

| Model | Source | Instructions |
|-------|--------|--------------|
| limubert | [LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public) | Download and place as `pretrained/limubert.pt` |
| imumae | [IMU-Video-MAE](https://github.com/mf-zhang/IMU-Video-MAE) | Download and place as `pretrained/imumae.pth` |
| selfpab | [SelfPAB](https://github.com/ntnu-ai-lab/SelfPAB) | Download and place as `pretrained/selfpab.ckpt` |

See `.url` files in `pretrained/` for download links.

## Citation

```bibtex
@inproceedings{harbench2026,
  title={HARBench: A Comprehensive Benchmark for Evaluating Foundation Models in Sensor-based Human Activity Recognition},
  author={Tanigaki, Kei and Maekawa, Takuya and Hara, Takahiro},
  booktitle={2026 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  year={2026}
}
```
