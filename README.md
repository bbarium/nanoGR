## GRBenchmark

本项目用于生成式推荐（Generative Recommendation）实验，主要包含三类流程：

- **train_sid.py**：将item表示映射为Semantic ID。
- **custom_finetune.py**：从头训练轻量级encoder-decoder生成推荐模型。
- **finetune.py**：在预训练大语言模型基础上扩展SID token，并进行生成推荐任务微调。

## 路径配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_path` | `./data/AmazonReviews` | Amazon Reviews数据集根目录 |
| `--gr_path` | `./checkpoints/Qwen3-0.6B` | `finetune.py`中用于对齐训练的预训练大模型路径 |
| `--output_path` | `./outputs` | SID文件、缓存文件、训练输出等默认保存目录 |
| `--log_path` | `./logs` | 日志文件保存目录 |
| `--projector_path` | `./checkpoints/sentence-t5-base` | `train_sid.py` 中用于生成 item 文本向量的句向量模型路径 |
| `--sem_ids_path` | `./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids` | 已训练好的item-to-SID映射文件路径 |
| `--config_path` | `./configs/enc_dec_gr_t5.yaml` | Encoder-decoder模型结构配置文件 |

推荐的数据与模型目录结构如下：

```text
GRBenchmark/
├── data/
│   └── AmazonReviews/
├── checkpoints/
│   ├── sentence-t5-base/
│   └── Qwen3-0.6B/
├── outputs/
├── logs/
└── configs/
```

其中：

- `./data/AmazonReviews` 放置原始推荐数据。
- `./checkpoints/sentence-t5-base` 放置sentence embedding模型，用于生成item表示。
- `./checkpoints/Qwen3-0.6B` 放置大语言模型checkpoint，用于大模型生成推荐对齐训练。
- `./outputs` 保存SID文件、句向量文件、训练缓存和模型输出。
- `./logs` 保存训练日志。

所有路径参数都可以在命令行中覆盖，例如：

```bash
python train_sid.py \
    --data_path ./data/AmazonReviews \
    --projector_path ./checkpoints/sentence-t5-base \
    --output_path ./outputs
```

## 训练脚本说明

### `train_sid.py`

`train_sid.py` 用于将 item 训练为 Semantic ID（SID），是后续生成推荐训练的前置步骤。

主要功能：

- 从 `--data_path` 读取原始 Amazon Reviews 数据。
- 使用 `--projector_path` 指定的句向量模型对 item metadata 进行编码。
- 将 item embedding 转换为离散 SID。
- 支持 OPQ 和 RQ-VAE 两类 SID 生成方法；当前默认训练流程使用 `RQVAETokenizer`。
- 输出 item-to-SID 映射，供 `custom_finetune.py` 和 `finetune.py` 使用。

关键参数：

| 参数 | 说明 |
|---|---|
| `--data_path` | 原始数据集路径，默认 `./data/AmazonReviews`。 |
| `--output_path` | SID 与中间结果输出路径，默认 `./outputs`。 |
| `--projector_path` | 句向量模型路径，默认 `./checkpoints/sentence-t5-base`。 |
| `--category` | 数据集类别，例如 `Toys_and_Games`、`Beauty`、`Sports_and_Outdoors`。 |
| `--metadata` | 使用的 metadata 类型，默认 `sentence`。 |
| `--num_codebooks` | SID 层级数量。 |
| `--codebook_size` | 每个 codebook 的大小。 |
| `--hid_channels`、`--mlp_channels` | RQ-VAE tokenizer 的网络结构参数。 |
| `--kmeans_init`、`--kmeans_iters` | 是否使用 k-means 初始化及其迭代次数。 |

示例：

```bash
python train_sid.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --projector_path ./checkpoints/sentence-t5-base \
    --output_path ./outputs \
    --num_codebooks 3 \
    --codebook_size 32
```

### `custom_finetune.py`

`custom_finetune.py` 用于从头训练一个 encoder-decoder 生成推荐模型。

该脚本不依赖预训练大语言模型，而是根据配置文件构建 encoder 和 decoder，并在 SID 序列上进行生成式推荐训练。

主要功能：

- 读取原始用户行为序列数据。
- 读取 `--sem_ids_path` 中的 item-to-SID 映射。
- 将用户历史 item 序列转换为 SID token 序列。
- 构建 encoder-decoder 生成推荐模型 `SemanticIDEncoderDecoder`。
- 支持单卡训练和 `torchrun` 分布式训练。
- 支持数据集 tokenization cache，加速重复实验。
- 在验证与测试阶段基于生成 SID 计算推荐指标。

关键参数：

| 参数 | 说明 |
|---|---|
| `--data_path` | 原始数据集路径，默认 `./data/AmazonReviews`。 |
| `--sem_ids_path` | SID 映射文件路径。 |
| `--output_path` | 训练输出和缓存目录，默认 `./outputs`。 |
| `--log_path` | 日志目录，默认 `./logs`。 |
| `--config_path` | 模型结构配置文件路径。 |
| `--category` | 数据集类别。 |
| `--training_window` | 训练样本构造方式，支持 `sliding` 和 `last_only`。 |
| `--num_codebooks` | SID 层级数量，需要与 SID 文件一致。 |
| `--codebook_size` | 每层 SID codebook 大小。 |
| `--max_seq_length` | 最大用户历史长度。 |
| `--num_beams` | beam search 数量。 |
| `--eval_topks` | 评估 top-k 列表，例如 `5,10`。 |
| `--should_check_prefix` | 是否在生成时检查 SID 前缀合法性。 |

单卡示例：

```bash
python custom_finetune.py \
    --config_path ./configs/enc_dec_gr_t5.yaml \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --output_path ./outputs \
    --log_path ./logs
```

多卡训练可以使用项目中的启动脚本：

```bash
bash finetune_enc_dec_gr.sh 4
```

也可以通过环境变量覆盖默认配置：

```bash
CATEGORY=Sports_and_Outdoors \
SEM_IDS_PATH=./outputs/sentence-t5-base_Sports_and_Outdoors_RQVAE3x256.dim32.sem_ids \
CONFIG_PATH=./configs/enc_dec_gr_t5.yaml \
PER_DEVICE_BS=1024 \
EVAL_BS=1024 \
bash finetune_enc_dec_gr.sh 4
```

### `finetune.py`

`finetune.py` 用于训练一个大模型对齐的生成推荐模型。

该脚本以预训练大语言模型为 backbone，通过扩展 tokenizer 中的 SID token，将推荐任务转换为文本生成/序列生成任务，并进行 supervised fine-tuning。

主要功能：

- 从 `--data_path` 读取原始用户行为序列数据。
- 从 `--sem_ids_path` 读取 item-to-SID 映射。
- 将 SID 扩展为大模型 tokenizer 中的新增 token。
- 使用 `AutoModelForCausalLM` 加载 `--gr_path` 指定的大语言模型。
- 构造文本化推荐训练样本和 dataloader。
- 使用 HuggingFace `Trainer` 完成大模型对齐训练。

关键参数：

| 参数 | 说明 |
|---|---|
| `--data_path` | 原始数据集路径，默认 `./data/AmazonReviews`。 |
| `--gr_path` | 预训练大模型路径，默认 `./checkpoints/Qwen3-0.6B`。 |
| `--sem_ids_path` | SID 映射文件路径。 |
| `--output_path` | HuggingFace Trainer 输出目录。 |
| `--category` | 数据集类别。 |
| `--num_codebooks` | SID 层级数量。 |
| `--codebook_size` | 每层 SID codebook 大小。 |
| `--max_seq_length` | 最大用户历史长度。 |
| `--train_from_scratch` | 若启用，则只从模型 config 初始化，而不加载预训练权重。 |
| `--enable_torch_compile` | 是否启用 `torch.compile`。 |

示例：

```bash
python finetune.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --gr_path ./checkpoints/Qwen3-0.6B \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --output_path ./outputs
```

## 推荐运行流程

通常建议按以下顺序运行：

1. **训练 SID**

```bash
python train_sid.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --projector_path ./checkpoints/sentence-t5-base \
    --output_path ./outputs
```

2. **训练生成推荐模型**

如果希望从头训练 encoder-decoder 生成推荐模型：

```bash
python custom_finetune.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --config_path ./configs/enc_dec_gr_t5.yaml \
    --output_path ./outputs
```

如果希望训练大模型对齐的生成推荐模型：

```bash
python finetune.py \
    --category Toys_and_Games \
    --data_path ./data/AmazonReviews \
    --gr_path ./checkpoints/Qwen3-0.6B \
    --sem_ids_path ./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids \
    --output_path ./outputs
```

## 注意事项

- `--sem_ids_path` 对应的 SID 文件必须与 `--category`、`--num_codebooks`、`--codebook_size` 保持一致。
- 若修改了 SID 文件、数据划分方式或最大历史长度，建议在运行 `custom_finetune.py` 时加上 `--rebuild_dataset_cache` 以重建缓存。
- `finetune.py` 会向大模型 tokenizer 中添加 SID token，因此训练输出目录中保存的是扩展 tokenizer 后的模型。
- 多卡训练 `custom_finetune.py` 时推荐使用 `finetune_enc_dec_gr.sh`，该脚本会自动选择可用端口，并支持通过环境变量覆盖类别、SID 路径、配置文件和 batch size。