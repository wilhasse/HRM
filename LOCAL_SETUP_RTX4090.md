# HRM Local Setup Guide for RTX 4090

This guide will help you set up and run the Hierarchical Reasoning Model (HRM) on your local machine with an RTX 4090 GPU.

## Prerequisites

- Ubuntu/Linux system (tested on Ubuntu 22.04)
- NVIDIA RTX 4090 GPU
- Python 3.8 or higher
- At least 32GB RAM recommended
- ~50GB free disk space

## Step 1: Install CUDA and GPU Drivers

### Check current CUDA version
```bash
nvidia-smi
```

If CUDA 12.6 is not installed, follow these steps:

```bash
# Download and install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

# Set CUDA environment variables
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

## Step 2: Create Python Environment

```bash
# Create conda environment (recommended)
conda create -n hrm python=3.10
conda activate hrm

# Or use venv
python3 -m venv hrm_env
source hrm_env/bin/activate
```

## Step 3: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.6 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Verify PyTorch can see your GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Step 4: Install FlashAttention

RTX 4090 uses the Ada Lovelace architecture, so install FlashAttention 2:

```bash
# Install build dependencies
pip3 install packaging ninja wheel setuptools setuptools-scm

# Install FlashAttention 2
pip3 install flash-attn --no-build-isolation
```

## Step 5: Clone Repository and Install Dependencies

```bash
# Clone the repository
git clone <repository-url> HRM
cd HRM

# Initialize submodules
git submodule update --init --recursive

# Install Python dependencies
pip install -r requirements.txt
```

## Step 6: Setup Weights & Biases (Optional but Recommended)

```bash
# Create W&B account at https://wandb.ai/
# Then login
wandb login
```

To run without W&B:
```bash
export WANDB_MODE=offline
```

## Step 7: Quick Demo - Sudoku Solver

### Build the dataset
```bash
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 \
    --num-aug 1000
```

### Start training
```bash
OMP_NUM_THREADS=8 python pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

**Expected runtime on RTX 4090**: ~6-8 hours (faster than RTX 4070's 10 hours)

## Step 8: Monitor Training

- If using W&B, check your dashboard at https://wandb.ai/
- Look for `eval/exact_accuracy` metric
- Training is successful when accuracy approaches 100%

## Performance Optimization for RTX 4090

### Memory Optimization
```bash
# Increase batch size for better GPU utilization
# RTX 4090 has 24GB VRAM (vs 12GB on RTX 4070)
OMP_NUM_THREADS=8 python pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=768 \  # Doubled from 384
    lr=1e-4 \  # Slightly higher LR for larger batch
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

### Power and Thermal Management
```bash
# Set power limit to prevent throttling (optional)
sudo nvidia-smi -pl 450  # RTX 4090 default is 450W

# Monitor GPU temperature and utilization
watch -n 1 nvidia-smi
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `global_batch_size` (try 256 or 192)
- Close other GPU-consuming applications
- Clear GPU cache: `python -c "import torch; torch.cuda.empty_cache()"`

### FlashAttention Build Errors
```bash
# If FlashAttention fails to build, try:
pip uninstall flash-attn
pip install flash-attn --no-build-isolation -v
```

### Slow Training Speed
- Ensure GPU is not thermal throttling: check `nvidia-smi` for temperature
- Verify PCIe bandwidth: `nvidia-smi -q | grep -i pcie`
- Check CPU isn't bottlenecking: use `htop` during training

### Import Errors
```bash
# If adam-atan2 fails to import
pip install git+https://github.com/evanatyourservice/adam-atan2-pytorch.git
```

## Running Other Experiments

### ARC-AGI (1000 examples)
```bash
# Build dataset
python dataset/build_arc_dataset.py

# Train (single GPU, ~20-24 hours)
OMP_NUM_THREADS=8 python pretrain.py \
    data_path=data/arc-aug-1000 \
    global_batch_size=96 \
    epochs=100000 \
    eval_interval=10000
```

### Maze Solving
```bash
# Build dataset
python dataset/build_maze_dataset.py

# Train (single GPU, ~1 hour)
OMP_NUM_THREADS=8 python pretrain.py \
    data_path=data/maze-30x30-hard-1k \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=384 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

## Using Pre-trained Checkpoints

```bash
# Download checkpoint (example: Sudoku)
huggingface-cli download sapientinc/HRM-checkpoint-sudoku-extreme --local-dir checkpoints/sudoku

# Evaluate
OMP_NUM_THREADS=8 python evaluate.py checkpoint=checkpoints/sudoku/checkpoint.pt
```

## Tips for Success

1. **Start with Sudoku**: It's the fastest experiment to verify your setup works
2. **Monitor early**: Check training logs in first 100 steps to catch issues early
3. **Save checkpoints**: Enable `checkpoint_every_eval=True` for long runs
4. **Temperature management**: Ensure good airflow around your RTX 4090
5. **Batch size tuning**: RTX 4090 can handle larger batches than suggested defaults

## Next Steps

- Visualize puzzles: Open `puzzle_visualizer.html` and upload your dataset folder
- For ARC evaluation: Use `arc_eval.ipynb` notebook after training
- Experiment with hyperparameters using Hydra overrides

Happy training! 🚀