=======
# Personalized Influence Maximization (PIM) 

## Overview

This project implements an encoder-decoder architecture:
- **Encoder**: Diffusion-Aware Transformer (DAT) for learning graph representations
- **Decoder**: Decision Transformer (DT) for sequential decision-making

The model learns to select seed nodes conditioned on the target spread and budget, enabling personalized influence maximization strategies.

## Architecture

### Encoder: Graph Representation Learning

1. **Graph Attention Network (GAT)**: Initial node feature transformation
2. **Laplacian Positional Encoding**: Adds structural information to node embeddings
3. **Graph Transformer Layers**: Multi-layer message passing with attention mechanisms
4. **Output**: Node embeddings `h`, edge embeddings `e`, and edge indices

**Key Files**:
- `Encoder/main.py`: Graph encoding pipeline
- `Encoder/DAT/models.py`: GraphTransformerNet implementation

### Decoder: Sequential Decision Making

The decoder uses either:

1. **Decision Transformer (DT)**: GPT based architecture that models sequences of (Return-to-Go, State, Action) tuples

**Key Features**:
- **Return-to-Go (RTG)**: Personalized conditioning mechanism
  - Initial RTG = Target Spread (TS)
  - Updated during episode: `RTG_t = RTG_{t-1} - reward_t`
- **State Representation**: Binary vector of size `N` (number of nodes), where `1` indicates selected node
- **Action Representation**: Probability distribution over nodes (softmax output)

**Key Files**:
- `Decoder/dt/models/decision_transformer1.py`: Decision Transformer implementation
- `Decoder/dt/evaluation/evaluate_episodes.py`: Evaluation functions

### Environment: Influence Simulation

**Key Files**:
- `Decoder/dt/evaluation/environment.py`: Environment class
- `Decoder/utils/graph_utils.py`: Influence computation utilities

## Installation

### Prerequisites

- Python 3.8.5
- CUDA 10.x (for GPU acceleration, optional but recommended)
- Conda (recommended) or pip

### Method 1: Using Conda (Recommended)

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AGIG
```

#### Step 2: Create Conda Environment

```bash
conda env create -f conda_env.yml
conda activate AGIG
```

#### Step 3: Install PyTorch Geometric and Additional Dependencies

```bash
# Install PyTorch Geometric dependencies (for PyTorch 1.8.1 and CUDA 10.2)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip install torch-geometric

# Install additional required packages
pip install scipy networkx
```

### Method 2: Using pip Only

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd AGIG
```

#### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install PyTorch

```bash
# For CUDA 10.2 (GPU support)
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU-only
pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
```

#### Step 4: Install PyTorch Geometric Dependencies

```bash
# For CUDA 10.2
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip install torch-geometric

# For CPU-only, use:
# pip install torch-geometric
```

#### Step 5: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` file contains the core dependencies. PyTorch and PyTorch Geometric must be installed separately due to platform-specific requirements.

## Dependencies

See `requirements.txt` for the complete list of dependencies.

### Core Dependencies

- **PyTorch** (1.8.1): Deep learning framework
  - Install with CUDA: `pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`
  - CPU-only: `pip install torch==1.8.1`
- **Transformers** (4.5.1): Hugging Face transformers library (GPT-2 base for Decision Transformer)
- **PyTorch Geometric** (≥1.7.0): Graph neural network library
  - Requires: `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`
- **NumPy** (1.20.3): Numerical computing
- **SciPy** (≥1.7.0): Sparse matrix operations
- **NetworkX** (≥2.6): Graph utilities

### Installation Order

1. Install PyTorch (with or without CUDA)
2. Install PyTorch Geometric dependencies:
   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
   ```
3. Install PyTorch Geometric:
   ```bash
   pip install torch-geometric
   ```
4. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies

- **wandb** (0.9.1): Experiment tracking (optional, not used in current code)
- **matplotlib**: For visualization (optional)
- **tqdm**: For progress bars (optional)

## Dataset Preparation

### Supported Datasets

The framework supports the following datasets:

1. **Cora**: Citation network
2. **Citeseer**: Citation network
3. **Facebook Food**: Social network
4. **Facebook TV Show**: Social network
5. **Friendster**: Large-scale social network
6. **Jazz**: Collaboration network
7. **Network**: General network dataset
8. **Wiki**: Wikipedia network
9. **YouTube**: Social network

### Dataset Format

Datasets should be placed in `Decoder/data/{dataset_name}/` with the following structure:

```
Decoder/data/
├── cora/
│   ├── cora.SG                    # Graph file (pickle format)
│   ├── cora-{method}-{budget}.pkl # Trajectory data
│   └── ...
├── citeseer/
│   ├── citeseer.SG
│   ├── citeseer-{method}-{budget}.pkl
│   └── ...
└── ...
```

**Important**: The code expects specific file naming conventions:
- Graph files: `{dataset_name}.SG` (e.g., `cora.SG`, `jazz.SG`)
- Trajectory files: `{dataset_name}-{method}-{budget}.pkl` (e.g., `cora-LT-5.pkl` for budget=0.05)
- Budget is stored as integer percentage: `budget * 100` (e.g., 0.05 → 5, 0.1 → 10)

**Special cases**:
- `friendster`: Uses `com-friendster.ungraph.txt.gz` instead of `.SG`
- `youtube`: Uses `com-youtube.ungraph.txt.gz` instead of `.SG`

### Graph File Format (`.SG`)

The graph file should be a pickle file (`.pkl` or `.SG`) containing a dictionary with:
- `adj`: Sparse adjacency matrix (scipy.sparse format)

Example:
```python
import pickle
import scipy.sparse as sp

graph = {'adj': sparse_adjacency_matrix}
with open('cora.SG', 'wb') as f:
    pickle.dump(graph, f)
```

### Trajectory File Format (`.pkl`)

Trajectory files should contain a **list of dictionaries**, each representing one episode:

```python
trajectories = [
    {
        'states': np.array([...]),      # Binary state vectors (N nodes)
        'actions': np.array([...]),      # Action indices (selected nodes)
        'rewards': np.array([...]),      # Reward values (influence increments)
        'dones': np.array([...])        # Episode termination flags (or 'terminal')
    },
    # ... more episodes
]
```

Each dictionary should have:
- `states`: Array of state vectors (shape: `[episode_length, num_nodes]`), binary vectors where `1` indicates selected node
- `actions`: Array of action indices (shape: `[episode_length]`), indices of selected nodes
- `rewards`: Array of reward values (shape: `[episode_length]`), incremental influence spread
- `dones` or `terminal`: Array of boolean flags (shape: `[episode_length]`), indicates episode termination

### Generating Trajectory Data

If trajectory data is not available, you may need to generate it using expert algorithms (e.g., Greedy, CELF) or other influence maximization methods.

## Usage

### Basic Training

**Note**: The default `main.py` runs experiments for all dataset/method/budget combinations. To train on a specific configuration, you need to modify the code or use command-line arguments.

Train a Decision Transformer model (modify `main.py` to run a single experiment):

```bash
python main.py \
    --env IM \
    --model_type dt \
    --K 20 \
    --batch_size 64 \
    --embed_dim 128 \
    --n_layer 3 \
    --learning_rate 1e-4 \
    --max_iters 10 \
    --num_steps_per_iter 1000 \
    --device cuda
```

**Note**: The `--dataset`, `--method`, and `--budget` arguments are not directly supported via command line in the current implementation. The code iterates through all combinations by default. To train on a specific configuration, modify the loop at the end of `main.py`:

```python
# In main.py, replace the loop with:
for dataset in ['cora']:  # Single dataset
    for method in ['LT']:  # Single method
        for budget in [0.1]:  # Single budget
            experiment(f'{dataset}_PIM', variant=vars(args), dataset=dataset, method=method, budget=budget)
```

### Training with Custom Parameters

```bash
python main.py \
    --env IM \
    --model_type dt \
    --dataset jazz \
    --method LT \
    --K 30 \
    --batch_size 128 \
    --embed_dim 256 \
    --n_layer 4 \
    --n_head 2 \
    --dropout 0.2 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --warmup_steps 20 \
    --max_iters 20 \
    --num_steps_per_iter 2000 \
    --num_eval_episodes 10 \
    --device cuda
```


### Batch Training (Multiple Datasets/Methods/Budgets)

The default `main.py` script runs experiments for **all combinations**:

```bash
python main.py
```

This will train models for:
- **Datasets**: `cora`, `citeseer`, `fb_food`, `fb_tvshow`, `friendster`, `jazz`, `wiki`, `youtube`
- **Methods**: `LT`, `IC`, `SIS`, `ICI`
- **Budgets**: `0.05`, `0.1`, `0.2`, `0.3`, `0.4` (as percentage of nodes)

**Warning**: This will run **8 × 4 × 5 = 160 experiments**, which may take a very long time. Consider modifying the loop in `main.py` to run specific combinations.

**Output**: Results are written to `{dataset}_{method}_{budget}_output.txt` files in the root directory.

## Training and Evaluation

### Training Process

1. **Data Loading**:
   - Load graph structure from `.SG` file
   - Load expert trajectories from `.pkl` file
   - Normalize states and compute statistics

2. **Batch Generation**:
   - Sample random trajectory segments
   - Compute Return-to-Go (RTG) using discounted cumulative sum
   - Pad sequences to fixed length `K`
   - Apply attention masks to ignore padding

3. **Forward Pass**:
   - Embed states, actions, RTG, and timesteps
   - Pass through transformer layers
   - Predict next action

4. **Loss Computation**:
   - MSE loss between predicted and target actions
   - Apply attention mask to filter padding
   - Backpropagate and update parameters

### Evaluation Process

1. **Environment Reset**:
   - Initialize state (all zeros)
   - Set target return (RTG) based on TS
   - Reset reward history

2. **Episode Loop** (up to `budget` steps):
   - Get action from model (conditioned on RTG)
   - Select node with highest probability (not already selected)
   - Update state and compute reward
   - Update RTG: `RTG = RTG - reward`
   - Check if budget reached

3. **Influence Computation**:
   - Use MC or RR method to estimate influence spread
   - RR sets are cached for efficiency
   - Compute incremental reward

### Metrics

The evaluation outputs:
- **Return Mean/Std**: Average episode return and standard deviation
- **Length Mean/Std**: Average episode length and standard deviation
- **Influence Spread (REWARD)**: Final influence spread from rewards
- **Influence Spread (R)**: Final influence spread from environment

## Project Structure

```
AGIG/
├── main.py                          # Main entry point for training and evaluation
├── requirements.txt                 # Python package dependencies (pip)
├── conda_env.yml                    # Conda environment configuration
├── README.md                        # This file
│
├── Encoder/                         # Graph encoding module
│   ├── main.py                      # Graph encoding pipeline (MyGT_gen, MyGT_train)
│   ├── graph.py                     # Graph utilities and encoding functions
│   ├── DAT/                         # Diffusion-Aware Transformer
│   │   ├── models.py                # GraphTransformerNet implementation
│   │   └── layers.py                # GraphTransformerLayer and MLPReadout
│   ├── GT/                          # Graph Transformer (alternative implementation)
│   │   ├── models.py                # Alternative GraphTransformerNet
│   │   └── layers.py                # Graph transformer layers
│   └── Data/                        # Graph data files (for testing)
│       └── twitter/                 # Twitter dataset (if available)
│
├── Decoder/                         # Sequential decision module
│   ├── dt/
│   │   ├── models/
│   │   │   ├── decision_transformer1.py  # Decision Transformer (GPT-2 based)
│   │   │   └── mlp_bc.py                 # Behavior Cloning model
│   │   ├── training/
│   │   │   ├── trainer.py           # Base trainer class
│   │   │   ├── seq_trainer.py       # Decision Transformer trainer
│   │   │   └── act_trainer.py       # Behavior Cloning trainer
│   │   └── evaluation/
│   │       ├── environment.py       # Influence simulation environment
│   │       ├── evaluate_episodes.py # Episode evaluation functions
│   │       └── models.py             # Additional model utilities
│   ├── utils/
│   │   └── graph_utils.py           # Graph utilities and influence computation
│   └── data/                        # Dataset files (REQUIRED)
│       ├── citeseer/             
│       ├── jazz/
│       ├── fb_tvshow/
│       ├── wiki/
│       └── youtube/
│
└── {dataset}_{method}_{budget}_output.txt  # Output files (generated during training)
│
└── MyDT3/                           # Alternative implementation (legacy)
    └── ...
```

## Configuration Parameters

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--env` | str | `'IM'` | Environment name (Influence Maximization) |
| `--mode` | str | `'normal'` | Training mode (`'normal'`, `'delayed'`, `'noise'`) |
| `--K` | int | `20` | Context window length (max sequence length) |
| `--pct_traj` | float | `1.0` | Percentage of trajectories to use for training |
| `--batch_size` | int | `64` | Batch size for training |
| `--model_type` | str | `'dt'` | Model type (`'dt'` or `'bc'`) |
| `--embed_dim` | int | `128` | Embedding dimension (hidden size) |
| `--n_layer` | int | `3` | Number of transformer layers |
| `--n_head` | int | `1` | Number of attention heads |
| `--activation_function` | str | `'relu'` | Activation function |
| `--dropout` | float | `0.1` | Dropout rate |
| `--learning_rate` / `-lr` | float | `1e-4` | Learning rate |
| `--weight_decay` / `-wd` | float | `1e-4` | Weight decay for optimizer |
| `--warmup_steps` | int | `10` | Number of warmup steps for LR scheduler |
| `--num_eval_episodes` | int | `6` | Number of episodes for evaluation |
| `--max_iters` | int | `1` | Maximum number of training iterations |
| `--num_steps_per_iter` | int | `1000` | Number of training steps per iteration |
| `--device` | str | `'cuda'` | Device for computation (`'cuda'` or `'cpu'`) |

### Environment Parameters

The environment is initialized with:
- **Budget**: `int(budget × num_nodes)` - Maximum number of seed nodes
- **Method**: `'MC'` or `'RR'` - Influence computation method
- **use_cache**: `True` - Enable RR set caching for efficiency

### Model Hyperparameters

**Decision Transformer**:
- `max_length`: Context window (default: 20)
- `max_ep_len`: Maximum episode length (budget × num_nodes)
- `hidden_size`: Embedding dimension
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_inner`: Inner dimension (4 × hidden_size)
- `resid_pdrop`: Residual dropout
- `attn_pdrop`: Attention dropout.

## Output Format

Training and evaluation results are written to:

```
{dataset}_{method}_{budget}_output.txt
```

Example: `cora_LT_0.1_output.txt`

### Output Contents

Each iteration outputs:
- Training diagnostics (action error, loss)
- Evaluation metrics:
  - `target_{TS}_return_mean`: Average episode return
  - `target_{TS}_return_std`: Standard deviation of returns
  - `target_{TS}_length_mean`: Average episode length
  - `target_{TS}_length_std`: Standard deviation of lengths
  - `dataset_{dataset}_diffusion_model_{method}_budget_{budget}_influence_spread_REWARD`: Influence spread from rewards
  - `dataset_{dataset}_diffusion_model_{method}_budget_{budget}_influence_spread_R`: Final influence spread

### Example Output

```
================================================================================
Iteration 1
training/action_error: 0.0234
target_2000000_return_mean: 1250.5
target_2000000_return_std: 45.2
target_2000000_length_mean: 10.0
target_2000000_length_std: 0.0
dataset_cora_diffusion_model_LT_budget_0.1_influence_spread_REWARD: 0.625
dataset_cora_diffusion_model_LT_budget_0.1_influence_spread_R: 0.630
time/total: 125.3
time/evaluation: 12.5
================================================================================
```

