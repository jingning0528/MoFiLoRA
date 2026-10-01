# MoFiLoRA: Momentum-Enhanced LoRA for Federated Recommendation

MoFiLoRA is a research implementation of communication-efficient federated recommendation using low-rank adaptation (LoRA) and momentum-based aggregation.

The project evaluates full-embedding, LoRA, fixed-factor, and momentum-enhanced variants across recommendation datasets while keeping the experiment pipeline configurable and reproducible.

## Highlights

- Low-rank item-embedding updates for federated neural collaborative filtering
- Momentum aggregation for LoRA factors
- Fixed-`A`, fixed-`B`, and combined momentum/fixed-factor ablations
- Configurable CPU and CUDA execution
- Evaluation with MRR, NDCG, and Hit Rate at multiple cutoffs
- Analysis and visualization scripts for convergence, communication, and ablation studies

## Models

The `zoo/` directory contains the primary implementations:

- `Full` and `Full_Mom` — full item-embedding baselines
- `LoRA` — low-rank federated recommendation baseline
- `LoRA_FixedA` and `LoRA_FixedB` — fixed-factor ablations
- `LoRA_MomA`, `LoRA_MomB`, and `LoRA_MomAB` — momentum variants
- `LoRA_MomA_FixedB` and `LoRA_MomB_FixedA` — combined variants used by MoFiLoRA experiments
- `DoRA` — decomposition-based comparison model

Earlier experimental implementations are retained in `zoo_v1/` and `zoo_v2/`.

## Datasets

The experiments use public recommendation datasets:

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- [Amazon Review Data](https://nijianmo.github.io/amazon/index.html)

Dataset definitions and local paths are configured in [`config/dataset_config.yaml`](config/dataset_config.yaml). Update its paths before running an experiment on a new machine.

Preprocessing scripts are available in `data_pre_process/`.

## Installation

Create a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Experiment settings are defined in [`config/model_config.yaml`](config/model_config.yaml). Each top-level key is an experiment ID containing its dataset, model, training, optimizer, and evaluation settings.

Important options include:

- `model` — model implementation exported by `zoo/`
- `dataset_id` — dataset entry from `config/dataset_config.yaml`
- `latent_dim` — LoRA rank
- `train_turn` — number of federated communication rounds
- `local_epoch` — local epochs per round
- `server_optimizer` — server update method, such as heavy-ball momentum
- `beta` and `eta_s` — momentum and server step-size parameters

## Running an Experiment

Run an experiment by passing its configuration ID:

```bash
python main.py --expid <experiment_id> --gpu <device>
```

Use `-1` for CPU or a CUDA device index such as `0` for GPU execution.

Example:

```bash
python main.py --expid ML1M --gpu 0
```

Override the configured random seed or model when needed:

```bash
python main.py \
  --expid Industrial \
  --gpu 0 \
  --seed 2025 \
  --model LoRA_MomA_FixedB
```

Add `--save_csv` to write a result summary. Use `--result_file` to choose its destination.

## Project Structure

```text
config/                # Dataset and experiment configuration
data_pre_process/      # MovieLens and Amazon preprocessing
dataloaders/           # Federated dataset loaders
framework/             # Training utilities and shared modules
zoo/                   # Current model implementations
zoo_v1/, zoo_v2/       # Earlier experimental variants
main.py                 # Experiment entry point
analyze_*.py            # Convergence and ablation analysis
figure*.py              # Plot generation
```

## Acknowledgements

The training framework is based on ideas and components from [FuxiCTR](https://github.com/reczoo/FuxiCTR).
