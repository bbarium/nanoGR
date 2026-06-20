## nanoGR

This project provides a training pipeline for Generative Recommendation (GR). The framework offers complete scripts for generating semantic IDs (SIDs) and supports both traditional GR training approaches and advanced LLM-integrated methods. The corresponding entry points are:

- **train_sid.py**: Map item representations to SIDs.
- **custom_finetune.py**: Train a lightweight encoder-decoder generative recommendation model from scratch.
- **finetune.py**: Extend a pretrained large language model with SID tokens and fine-tune it on the generative recommendation task.

## Path Configuration

| Argument | Description |
|---|---|
| `--data_path`      | Root directory of the Amazon Reviews dataset |
| `--gr_path`        | Path of the pretrained large language model  |
| `--output_path`    | Default directory for saving SID files, cache files, training outputs, etc. |
| `--log_path`       | Directory for saving log files |
| `--projector_path` | Path of the sentence embedding model to generate item text embeddings |
| `--sem_ids_path`   | Path of the trained item-to-SID mapping file |
| `--config_path`    | Configuration file describing the encoder-decoder model architecture |

## Recommended Workflow

The recommended order to run the pipeline is:

### 1. Train SIDs

`train_sid.py` is used to convert item representations into SIDs, which is a prerequisite step for the subsequent generative recommendation training.

**Example:**

```bash
python train_sid.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --projector_path ./checkpoints/sentence-t5-base \
    --output_path ./outputs
```

### 2. Train the Generative Recommendation Model

#### Path A — Train an encoder-decoder GR model from scratch

Use this path when you want a **lightweight, self-contained** GR model that does not depend on any pretrained LLM. `custom_finetune.py` builds an encoder–decoder according to a YAML config and trains it directly on SID sequences.

**Single-GPU Example:**

```bash
python custom_finetune.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --config_path ./configs/enc_dec_gr_t5.yaml \
    --output_path ./outputs
```

**Multi-GPU Example (via `torchrun`):**

```bash
torchrun --standalone --nproc_per_node=4 custom_finetune.py \
    --config_path ./configs/enc_dec_gr_t5.yaml \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --output_path ./outputs \
    --log_path ./logs \
    --per_device_batch_size 1024 \
    --eval_batch_size 1024
```

#### Path B — Train an LLM-aligned GR model

Use this path when you want to **leverage a pretrained large language model** as the backbone. `finetune.py` extends the LLM tokenizer with SID tokens, reformulates the recommendation task as sequence generation, and performs supervised fine-tuning.

**Example:**

```bash
python finetune.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --gr_path ./checkpoints/Qwen3-0.6B \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --output_path ./outputs
```