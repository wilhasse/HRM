# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Training Commands
- **Quick demo (Sudoku, single GPU)**: `OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0`
- **Full training (8 GPUs)**: `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py`
- **Evaluation**: `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>`

### Dataset Building
- **ARC-1**: `python dataset/build_arc_dataset.py`
- **ARC-2**: `python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000`
- **Sudoku**: `python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`
- **Maze**: `python dataset/build_maze_dataset.py`

### Environment Setup
- Initialize submodules: `git submodule update --init --recursive`
- Login to W&B: `wandb login`

## Architecture Overview

HRM (Hierarchical Reasoning Model) is a recurrent architecture for complex reasoning tasks with two main components:

1. **High-level module**: Handles slow, abstract planning (H_cycles, H_layers parameters)
2. **Low-level module**: Performs rapid, detailed computations (L_cycles, L_layers parameters)

Key architectural files:
- `models/hrm/hrm_act_v1.py`: Core HRM implementation with Adaptive Computation Time
- `models/losses.py`: ACT loss implementations (stablemax_cross_entropy, softmax_cross_entropy)
- `models/sparse_embedding.py`: Distributed sparse embedding for puzzle representations

## Configuration System

Uses Hydra for configuration management:
- Main config: `config/cfg_pretrain.yaml`
- Architecture config: `config/arch/hrm_v1.yaml`
- Override parameters via command line: `key=value` syntax

Key hyperparameters:
- `global_batch_size`: Total batch size across GPUs
- `lr`, `puzzle_emb_lr`: Learning rates for model and puzzle embeddings
- `H_cycles`, `L_cycles`: Number of recurrent cycles for high/low-level modules
- `halt_max_steps`: Maximum ACT steps (adaptive computation)

## Training Pipeline

1. **Data loading**: `PuzzleDataset` class handles ARC/Sudoku/Maze datasets
2. **Model**: Instantiated via `load_model_class()` using config string format
3. **Optimization**: AdamATan2 optimizer with cosine LR schedule
4. **Distributed**: Uses PyTorch DDP for multi-GPU training
5. **Logging**: W&B integration for experiment tracking (eval/exact_accuracy metric)

## Dataset Format

Puzzles stored as HDF5 files with structure:
- `puzzles/{puzzle_id}/metadata`: Puzzle metadata
- `puzzles/{puzzle_id}/examples/{idx}`: Input/output pairs
- Supports augmentation via `num_aug` parameter during dataset building