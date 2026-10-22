# nanoGR 实验启动指南

本文记录 Amazon Reviews 2014 Beauty 类别的生成式推荐实验，说明从环境准备、数据处理、SID 生成，到单卡训练和跨服务器断点恢复的完整操作。目标是先建立一个能运行、能保存模型、能输出推荐指标的实验基线。

基于原仓库提交 `6274cb47ce8b9e466bb17104561d1cef9f2a787c`（简称 `6274cb4`），结合 2026 年 10 月 21—22 日的实际运行记录整理。实验代码在服务器上已做运行修复；原始仓库直接运行仍会遇到文中列出的错误。

本文与仓库根目录的 `prepare_nanogr.py` 配套使用。脚本只复现本次已使用的代码修改，默认检查，显式添加 `--apply` 才写文件，并保留备份。服务器上的修复代码和实验结果不会因为加入本文就自动同步到本地或远程 Git 仓库。作者原始说明保留在 [docs/README.upstream.md](docs/README.upstream.md)。

## 1. 根据当前进度选择入口

| 当前情况 | 从哪里开始 |
|---|---|
| 新服务器，没有环境、数据和模型 | 第 3 节 |
| 环境已安装，数据和文本模型已准备好 | 第 5 节或第 6 节 |
| 已有 `Beauty` 的 `.sem_ids` 文件，只想开启推荐训练 | 第 7 节 |
| 已有推荐模型 `best_model.pt`，想恢复训练 | 第 9 节 |
| 从 rec2 迁移到 rec6 | 第 10 节 |
| 忘记怎样离开、恢复或停止 tmux 中的任务 | 第 8 节 |

所有训练命令都在 Linux 服务器上运行。本文以项目路径 `/home/bywu/nanoGR` 为例；其他账号请替换路径。服务器地址、密码和资源分配情况不写入本指南，登录时使用自己的账号信息。

## 2. 本次实验是什么

```text
物品标题、描述等元信息
          ↓ sentence-t5-base
       768 维物品向量
          ↓ RQ-VAE
       三位语义 ID（SID）
          ↓ 追加一位冲突区分编号
       推荐训练实际使用四位 SID

用户历史物品 → 历史 SID 序列 → 编码器—解码器 → 下一个物品的 SID
```

SID 生成和推荐训练分开进行。推荐训练阶段使用固定 SID，不会反向更新前面的文本模型或 RQ-VAE。推荐模型采用 T5 结构并从头训练，不是微调 Qwen；用于提取物品文本向量的 `sentence-t5-base` 则是预训练模型。

| 项目 | 本次设置 |
|---|---|
| 数据集 | Amazon Reviews 2014，Beauty 5-core 评论及类别元信息 |
| 真实用户 / 真实物品 / 交互 | 22,363 / 12,101 / 198,502 |
| 数据划分 | 每个用户最后一条用于测试，倒数第二条用于验证 |
| 训练样本构造 | `sliding`，从前面的序列构造多个历史—目标样本 |
| 训练 / 验证 / 测试样本 | 131,413 / 22,363 / 22,363 |
| 最大历史长度 | 20 个物品，不是 20 个 SID token |
| SID | RQ-VAE，3 层，每层 256 个码，隐空间 32 维 |
| 推荐模型 | T5 结构，编码器和解码器各 4 层，`d_model=128` |
| 推荐 batch size / 随机种子 | 64 / 42 |
| 优化器 / 学习率 / 权重衰减 | AdamW / 0.001 / 0.01 |
| 学习率调度 | cosine，warmup ratio=0.05，最低学习率比例=0.1 |
| 生成 | Beam Search，10 beams，开启 SID 前缀约束 |
| 最佳模型选择 | 验证集 `hit@5` |
| 评估间隔 / 早停 | 每 5 轮；连续 5 次评估没有改善则停止 |
| 训练轮数上限 | 总共 100 轮，可能提前停止 |

原训练器也在第 1 轮和最后一轮评估。早停中的 5 指评估次数，不是 5 个 epoch。当前训练器每次验证时也运行测试评估；选择配置和模型时仍应只依据验证集。

## 3. 服务器和代码准备

登录目标服务器后，先确认机器、资源和环境：

```bash
hostname
nvidia-smi
df -h .
conda env list
```

GPU 是否可用以实验室安排为准，`nvidia-smi` 只是当前使用快照。本文后续以物理 GPU 2 为例；rec2 只有 GPU 0、1，不能照用 `2`。

第一次获取代码：

```bash
cd /home/bywu
git clone https://github.com/Zhhuizhe/nanoGR.git
cd /home/bywu/nanoGR
git rev-parse --short HEAD
```

本文修复脚本针对 `6274cb4`。已有项目不要重复 clone；已有修改的项目不要为了匹配本指南强制重置版本。更新后的上游代码需要重新核对，不保证仍适用相同文本替换。

### 3.1 应用本次运行修复

将配套的 `prepare_nanogr.py` 放在项目根目录，先检查：

```bash
python prepare_nanogr.py
```

检查成功后写入修改：

```bash
python prepare_nanogr.py --apply
```

已存在相同修复时会显示 `OK`，不会重复插入。若代码不匹配，脚本在写入源码前退出；不要强行应用到不同版本。备份文件以 `.before_readme.bak` 结尾。

脚本包含以下修复：

| 文件 | 修改内容 |
|---|---|
| `train_sid.py` | 将 `epochs`、`eval_step`、`train_batch_size` 改为整数参数 |
| `nanogr/tokenization/rq_vae.py` | 恢复 SID JSON 保存，创建输出目录 |
| `custom_finetune.py` | 将不存在的 `args.eval_batch` 改为 `args.eval_batch_size` |
| `nanogr/dist_dataloader.py` | 支持 `from_tensors` 初始化，创建缓存目录，先定义 `item_seqs` 再计算长度 |
| `nanogr/trainer.py` | 支持通过 `NANOGR_RESUME_PATH` 恢复模型、优化器、调度器和最佳验证分数 |

脚本不会修复项目所有潜在问题，本文只覆盖已经实际跑通的单卡编码器—解码器路线。没有在这次实验中验证多卡、OPQ 或语言模型微调路线。

## 4. 安装和验证环境

本次验证过的主要版本：Python 3.10.22、PyTorch 2.6.0+cu124、NumPy 1.26.4、Transformers 4.56.2、sentence-transformers 5.1.1。硬件包括 TITAN RTX 和 RTX 4090；对应驱动为 570.172.08 和 550.163.01。

首次创建：

```bash
conda create -n nanogr python=3.10 -y
conda activate nanogr
python --version
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4
python -m pip install scikit-learn==1.7.2 scipy==1.15.3 pyyaml==6.0.3 logzero==1.7.0 tqdm==4.70.1 tensorboard==2.21.0
python -m pip install transformers==4.56.2 sentence-transformers==5.1.1 sentencepiece==0.2.2 huggingface-hub==0.36.2
```

这些是本次记录中的直接依赖版本，不是完整的传递依赖锁文件。环境创建好以后可保存完整清单：

```bash
mkdir -p logs
python -m pip freeze > logs/environment_freeze.txt
conda list --explicit > logs/conda_explicit.txt
```

已经有可用环境时直接 `conda activate nanogr`，不需要每次安装。也可以按第 10 节迁移已有环境，减少下载。

确认项目依赖可导入：

```bash
cd /home/bywu/nanoGR
python -c "from nanogr.dataset import AmazonReviews2014; from nanogr.tokenization import RQVAETokenizer; from nanogr.nn import SemanticIDEncoderDecoder; from sentence_transformers import SentenceTransformer; print('主要模块导入成功')"
```

确认一张获准使用的 GPU 能计算：

```bash
CUDA_VISIBLE_DEVICES=2 python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); x = torch.ones(3, device='cuda:0'); print('计算结果:', (x + x).cpu())"
```

物理 GPU 2 被指定为唯一可见卡后，程序内部的 `cuda:0` 就对应它。`nvidia-smi` 的 CUDA Version 表示驱动支持的版本上限，不等同于本地 CUDA Toolkit 版本。当前使用预编译 PyTorch 包的路线不需要额外安装系统 CUDA Toolkit。

## 5. 准备数据和文本向量模型

### 5.1 下载 Beauty 数据

优先检查实验室是否已有相同版本的数据。不要将处理缓存写进只读的公共目录；需要时在自己的目录内建立原始文件软链接。

本实验的可写目录布局：

```text
/home/bywu/nanoGR/
├── data/AmazonReviews/Beauty/
│   ├── raw/
│   │   ├── reviews_Beauty_5.json.gz
│   │   └── meta_Beauty.json.gz
│   └── processed/                 # 程序生成
├── checkpoints/sentence-t5-base/   # 文本向量模型
├── configs/enc_dec_gr_t5.yaml
├── outputs/                       # 向量、SID、推荐断点、数据缓存
├── logs/                          # 推荐日志和控制台日志
└── runs/                          # SID 训练 TensorBoard 日志
```

首次下载：

```bash
mkdir -p /home/bywu/nanoGR/data/AmazonReviews/Beauty/raw
cd /home/bywu/nanoGR/data/AmazonReviews/Beauty/raw
wget -c https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
wget -c https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
gzip -t reviews_Beauty_5.json.gz meta_Beauty.json.gz && echo "压缩文件完整性检查通过"
```

`-c` 支持续传；不要事先解压，代码直接读取 gzip 文件。`gzip -t` 检查文件能否完整解压，不是来源身份或内容版本验证。

预处理与数量检查：

```bash
cd /home/bywu/nanoGR
python -c "from nanogr.dataset import AmazonReviews2014; d = AmazonReviews2014('./data/AmazonReviews', 'Beauty', 'sentence'); print('真实用户数:', d.n_users - 1); print('真实物品数:', d.n_items - 1); print('交互数:', d.n_interactions)"
```

预期为 `22363`、`12101`、`198502`。减去 1 是排除编号映射中保留的 padding 项。生成 `processed/` 后，再运行会读取已有处理结果。

### 5.2 下载 sentence-t5-base

正常网络使用 Hugging Face。先确认当前终端没有遗留离线模式：

```bash
unset HF_HUB_OFFLINE
```

如果 Hugging Face 可访问：

```bash
export HF_ENDPOINT=https://huggingface.co
```

本次服务器直连出现 SSL 错误，改用镜像后成功；需要时将上面一行替换为：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

镜像是备用下载渠道，是否可达取决于当前网络。不要通过关闭 SSL 校验处理连接问题。

下载命令：

```bash
cd /home/bywu/nanoGR
export HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/sentence-t5-base', revision='fc5d4628481afbbaaacd7af6bb07cf9d3865f781', local_dir='./checkpoints/sentence-t5-base', allow_patterns=['*.json', '*.safetensors', '*.bin', '*.model', '*.txt'], max_workers=2); print('模型下载完成')"
```

这里固定了本次下载响应中显示的模型提交，减少上游后续更新带来的变化。模型文件已经从另一台服务器复制过来时可以跳过下载。

CPU 离线检查：

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('./checkpoints/sentence-t5-base', device='cpu', local_files_only=True); x = m.encode(['This is a beauty product.']); print('向量形状:', x.shape)"
```

预期输出 `(1, 768)`。

### 5.3 生成所有物品向量

```bash
mkdir -p outputs logs
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 python -c "import torch; from nanogr.dataset import AmazonReviews2014; from nanogr.encoder import encode_sent_to_emb; d = AmazonReviews2014('./data/AmazonReviews', 'Beauty', 'sentence'); x = encode_sent_to_emb(d, './checkpoints/sentence-t5-base', 64, torch.device('cuda:0')); print('物品向量形状:', x.shape)"
```

预期形状 `(12101, 768)`，缓存文件为 `outputs/Beauty.sentence-t5-base.sent_emb`。模型和数据已经在本地时使用 `HF_HUB_OFFLINE=1`，避免训练时再访问下载站点。

不要对不同文本模型复用这份向量缓存。现有代码的缓存文件名没有包含完整的模型标识，更换模型或元信息处理方式后，需要另行区分和重新生成缓存。

## 6. 训练和检查物品 SID

如果已经有 `outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids`，先跳到 6.2 检查，不要重新训练。

### 6.1 生成 SID

确保已运行第 3.1 节的修复脚本，且 `outputs/` 已创建。在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 python -u train_sid.py \
  --category Beauty \
  --data_path ./data/AmazonReviews \
  --projector_path ./checkpoints/sentence-t5-base \
  --num_codebooks 3 \
  --codebook_size 256 \
  --hid_channels 32 \
  --train_batch_size 4096 \
  --lr 0.0001 \
  --epochs 1000 \
  --eval_step 100 \
  --num_warmup_steps 10 \
  2>&1 | tee logs/beauty_sid_3x256_dim32.console.log
```

本次训练先在 rec2 的 GPU 1 执行；上面的 `2` 是为 rec6 编写的示例，迁移到其他卡上运行不保证得到逐位相同的 SID。

预期保存：

```text
outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids
```

文件虽然不是 `.json` 后缀，内容是 JSON 字典，键为原始物品 ID，值为三个整数。训练入口没有把 `--output_path` 传给所有下游组件，本指南因此统一使用默认的 `./outputs`，不要只改一个参数就假定全部文件会迁移。

原版加载已有 SID 的分支还存在返回变量未初始化的问题；配套脚本没有改变该分支。已有文件应直接用于推荐训练。如要重新训练，先保留现有 SID，再为新实验安排独立的文件路径；不要覆盖旧结果。

### 6.2 检查保存的 SID

```bash
python - <<'PY'
import json
from collections import Counter

with open('outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids') as f:
    data = json.load(f)
counts = Counter(tuple(sid) for sid in data.values())
print('物品数:', len(data))
print('不同 SID 数:', len(counts))
print('冲突率:', f'{1 - len(counts) / len(data):.2%}')
print('最大同码物品数:', max(counts.values()))
print('各层使用的编码数:', [len({sid[i] for sid in data.values()}) for i in range(3)])
PY
```

本次实际保存结果：

```text
物品数: 12101
不同 SID 数: 10266
冲突率: 15.16%
最大同码物品数: 37
各层使用的编码数: [5, 240, 231]
```

这一版本可以用于跑通推荐流程：同码最多 37 个物品，追加的一位编号有 256 种取值，足以区分。第一层仅使用了 5 个编码，编码质量仍有改善空间，不应将这份配置视为已经优化好的 SID 方法。

日志中的 `Collision Rate` 是 `1 - 不同 SID 数 / 物品数`，不是两两碰撞概率。训练日志里的利用率也不是最终全量物品使用编码数的直接替代品，应按上面脚本检查保存文件。

RQ-VAE 当前实现把同一批物品向量同时用于训练和所谓的验证，日志 `Val Loss` 不能作为独立验证集上的泛化证据。SID 保存条件混合使用损失与冲突率，保存文件也不一定来自最后一轮。推荐阶段的训练、验证、测试划分是另一层流程。

## 7. 启动推荐训练

### 7.1 启动前确认

```bash
cd /home/bywu/nanoGR
conda activate nanogr
nvidia-smi
ls configs/enc_dec_gr_t5.yaml
ls outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids
python prepare_nanogr.py
mkdir -p logs
```

检查脚本应全部为 `OK`。若显示 `NEEDS`，先按第 3.1 节应用修复。`--config_path` 必须显式传入现有的 `enc_dec_gr_t5.yaml`，原入口默认的另一配置文件不在这个版本的仓库中。

### 7.2 第一次先跑 1 轮

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 python -u custom_finetune.py \
  --category Beauty \
  --data_path ./data/AmazonReviews \
  --sem_ids_path ./outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids \
  --config_path ./configs/enc_dec_gr_t5.yaml \
  --output_path ./outputs \
  --log_path ./logs \
  --run_name beauty_smoke_seed42 \
  --seed 42 \
  --codebook_size 256 \
  --max_seq_length 20 \
  --per_device_batch_size 64 \
  --eval_batch_size 64 \
  --dataloader_num_workers 0 \
  --epochs 1 \
  --save_and_eval_step 1 \
  --enable_check_prefix \
  2>&1 | tee logs/beauty_smoke_seed42.console.log
```

应依次看到训练、验证、测试进度以及 `[TEST] Final metrics`。1 轮仅验证流程，不作为模型充分训练后的效果。

### 7.3 开始最多 100 轮的训练

长期运行前按第 8 节进入 tmux。在 tmux 内激活环境、进入项目目录后执行：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 python -u custom_finetune.py \
  --category Beauty \
  --data_path ./data/AmazonReviews \
  --sem_ids_path ./outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids \
  --config_path ./configs/enc_dec_gr_t5.yaml \
  --output_path ./outputs \
  --log_path ./logs \
  --run_name beauty_baseline_seed42 \
  --seed 42 \
  --codebook_size 256 \
  --max_seq_length 20 \
  --per_device_batch_size 64 \
  --eval_batch_size 64 \
  --dataloader_num_workers 0 \
  --epochs 100 \
  --save_and_eval_step 5 \
  --early_stop 5 \
  --enable_check_prefix \
  2>&1 | tee logs/beauty_baseline_seed42.console.log
```

这是从头训练，不会因为同名输出目录中已有模型就自动恢复。已有同名实验时使用新的 `--run_name`，同时修改 `tee` 的日志名；`tee` 默认会覆盖同名控制台日志，模型保存也会更新该目录的最佳模型。

`--enable_check_prefix` 开启真实物品 SID 的前缀约束。`--codebook_size` 必须与生成 SID 时使用的码本大小一致。程序自动追加冲突位，无需手动指定推荐模型的 SID 长度。

当前 `build_or_load_datasets()` 每次会重新构造并写入 `_dataset_cache/semid_Beauty.pt`，不是看到缓存就直接复用。不要让多个 Beauty 实验同时写入同一个输出目录的缓存；并行实验应使用不同的 `--output_path`。

## 8. tmux 和进程管理

在服务器登录终端创建会话：

```bash
tmux new -s nanogr_rec6
```

新会话内执行：

```bash
conda activate nanogr
cd /home/bywu/nanoGR
```

然后执行训练命令。

| 操作 | 作用 |
|---|---|
| `Ctrl+B`，松开后按 `D` | 离开 tmux；训练继续运行 |
| `tmux ls` | 查看当前账号的会话 |
| `tmux attach -t nanogr_rec6` | 返回已有会话 |
| 在训练前台按 `Ctrl+C` | 请求中断当前训练，不保证保存当轮 |
| 在训练前台按 `Ctrl+Z` | 暂停任务，通常仍占显存；不是后台继续训练 |
| 同一个 shell 执行 `jobs -l` | 查看这个 shell 的后台或暂停任务 |
| 同一个 shell 执行 `fg %1` | 将该 shell 的作业 1 恢复到前台 |

`%1` 是当前 shell 的作业号，换一个 SSH 终端通常看不到原来的作业表。始终先用 `jobs -l` 确认编号，不要照抄其他终端的作业号。

如果确认要结束自己的暂停作业 1：

```bash
jobs -l
kill -TERM %1
kill -CONT %1
```

第二个信号让暂停的进程继续处理退出请求。随后用 `jobs -l` 和 `nvidia-smi` 确认退出；不要结束其他人的 PID。停止训练不会删除之前保存的最佳 checkpoint。

从另一个登录终端监控：

```bash
nvidia-smi
tail -n 30 /home/bywu/nanoGR/logs/beauty_baseline_seed42.console.log
```

温度需要结合具体卡型、持续变化和实验室要求判断；80 多度不能单凭一个数字认定故障。可通过 `nvidia-smi -i 2 -q -d TEMPERATURE` 查看该卡报告的阈值。温度异常时先暂停自己的任务、检查和联系管理员；降低 batch size 不保证降低持续负载温度，功耗上限和散热设置由管理员协调。

## 9. 从 checkpoint 恢复训练

恢复功能是本次添加的，不是原仓库自带的参数。必须先应用配套修复脚本，入口是环境变量 `NANOGR_RESUME_PATH`，没有 `--resume_from_checkpoint` 参数。

### 9.1 检查断点

只对自己生成、可信的 checkpoint 使用下面的完整反序列化方式：

```bash
python - <<'PY'
import torch

path = 'outputs/beauty_baseline_seed42/best_model.pt'
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print('保存轮次:', ckpt['epoch'])
print('验证指标:', ckpt['metrics'])
print('保存内容:', list(ckpt.keys()))
PY
```

本次 rec2 迁移过来的断点保存于第 35 轮，包含：

```text
epoch, model_state_dict, optimizer_state_dict,
scheduler_state_dict, metrics, configs
```

`best_model.pt` 是按验证集指标保存的最佳模型，不一定是暂停前最后一轮的模型，也不是 `Ctrl+Z` 时进程内存的快照。第 35 轮断点从第 36 轮继续。

### 9.2 恢复命令

在 rec6 的 tmux 会话中执行：

```bash
mkdir -p logs
NANOGR_RESUME_PATH=/home/bywu/nanoGR/outputs/beauty_baseline_seed42/best_model.pt \
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
python -u custom_finetune.py \
  --category Beauty \
  --data_path ./data/AmazonReviews \
  --sem_ids_path ./outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids \
  --config_path ./configs/enc_dec_gr_t5.yaml \
  --output_path ./outputs \
  --log_path ./logs \
  --run_name beauty_rec6_resume_seed42 \
  --seed 42 \
  --codebook_size 256 \
  --max_seq_length 20 \
  --per_device_batch_size 64 \
  --eval_batch_size 64 \
  --dataloader_num_workers 0 \
  --epochs 100 \
  --save_and_eval_step 5 \
  --early_stop 5 \
  --enable_check_prefix \
  2>&1 | tee logs/beauty_rec6_resume_seed42.console.log
```

`--epochs 100` 是总轮数上限，不是再训练 100 轮。恢复时保持原来的数据、SID 文件、模型 YAML、历史长度、训练 batch size、梯度累积和调度器配置；不要随意改总轮数改变学习率日程。补丁会检查部分配置一致性，不会全面检查所有文件是否相同。

预期先输出 `[RESUME] 从第 36 轮继续训练`，再运行一次 `test-resume`，把继承的最佳模型保存到新输出目录，随后进入 `TRAIN epoch=36`。这份初始测试结果也用于后续没有更好验证模型时返回有效的最终指标。

恢复保留模型参数、优化器状态、学习率调度器状态和最佳验证分数。原 checkpoint 没有保存随机数状态；恢复时早停计数从 0 开始，未恢复的 DataLoader 随机顺序和跨 GPU 差异都可能改变后续轨迹，因此不承诺逐步精确复现原机器的连续训练。

如果要再次恢复，应把 `NANOGR_RESUME_PATH` 指向最新一次实验目录中的最佳模型，并使用新的输出名称，保留已有日志。

## 10. rec2 迁移到 rec6

迁移前先暂停或停止自己在源服务器上的训练，避免复制过程中 checkpoint 或日志变化。代码、数据、SID、模型和 Conda 环境是两类内容，分别复制；tmux 会话和内存中的运行进程不会一起迁移。

### 10.1 复制项目

目标目录尚不存在时，在 rec6 执行，替换连接地址占位符：

```bash
scp -r bywu@<rec2地址>:/home/bywu/nanoGR /home/bywu/
```

如果目标目录已经有实验结果，先安排独立接收目录再合并，不要重复照抄覆盖。迁移 SID 和 `best_model.pt` 后不需要重做 SID 训练。

### 10.2 复用环境

一般优先考虑 [conda-pack](https://conda.github.io/conda-pack/)；本次打包遇到 Conda 对 pip 的文件记录异常，尝试新版工具后仍出现 `python3.1` 路径问题，没有强行修改原环境元数据。

本次实际可用的替代方法：两台都是 Linux x86_64，环境的绝对路径完全一致，均为 `/home/bywu/anaconda3/envs/nanogr`，因此按原路径完整复制环境，然后验证。这个办法不是任意操作系统、目录或驱动之间都通用的迁移保证。

在 rec2 打包：

```bash
tar -czf /home/bywu/nanogr_env_samepath.tar.gz -C /home/bywu/anaconda3/envs nanogr
ls -lh /home/bywu/nanogr_env_samepath.tar.gz
```

不要覆盖已有的同名环境归档；需要保留时先改用新的归档名。本次归档约 3.0GB。

在 rec6 接收：

```bash
scp bywu@<rec2地址>:/home/bywu/nanogr_env_samepath.tar.gz /home/bywu/
```

如果 rec6 已经创建了一个未配置完的 `nanogr` 环境，先退出它，再保留为备份。本次使用以下步骤；仅当备份路径不存在时执行 `mv`，否则另选一个新备份名：

```bash
conda deactivate
test ! -e /home/bywu/nanogr_before_migration && \
  mv -T /home/bywu/anaconda3/envs/nanogr /home/bywu/nanogr_before_migration
```

确认原环境已被移开、目标 `envs/nanogr` 不存在后，才解压，避免把新旧环境混在一起：

```bash
tar -xzf /home/bywu/nanogr_env_samepath.tar.gz -C /home/bywu/anaconda3/envs
conda activate nanogr
python --version
CUDA_VISIBLE_DEVICES=2 python -c "import torch; from sentence_transformers import SentenceTransformer; print(torch.__version__); print(torch.cuda.get_device_name(0)); print((torch.ones(3, device='cuda:0') * 2).cpu())"
```

本次验证结果为 Python 3.10.22、PyTorch 2.6.0+cu124、RTX 4090，GPU 计算成功。只复制相同路径且验证通过的环境，不要直接把这份归档解压到另一个用户的任意路径。

确认目标机训练正常、日志和模型已保留后，再决定是否结束源服务器上自己的暂停任务。

## 11. 输出在哪里，如何判断实验完成

以恢复实验为例：

```text
logs/
├── beauty_rec6_resume_seed42.log          # 程序日志
└── beauty_rec6_resume_seed42.console.log  # tee 保存的终端输出
outputs/
├── Beauty.sentence-t5-base.sent_emb
├── sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids
├── _dataset_cache/semid_Beauty.pt
└── beauty_rec6_resume_seed42/
    ├── best_model.pt
    └── best_metrics.json
```

看到最后的 `[TEST] Final metrics: {...}`，并返回命令提示符，说明训练流程正常结束。可能是达到总轮数上限，也可能是早停；查看 `[TRAIN] Early stopped` 和最后的 epoch 才能判断。

最后一次 `[TEST] {...}` 是最后一次评估；`Final metrics` 对应验证集最佳模型时刻的测试指标，数值可以不同。`best_metrics.json` 保存的是验证指标，不是最终测试指标。当前代码没有单独保存最终测试 JSON，请保留控制台日志。

查看结果：

```bash
cat outputs/beauty_rec6_resume_seed42/best_metrics.json
grep 'Final metrics' logs/beauty_rec6_resume_seed42.console.log
grep -E 'Early stopped|\[TRAIN\] epoch=' logs/beauty_rec6_resume_seed42.console.log | tail -n 10
```

## 12. 本次实际结果记录

本次实验先在 rec2 训练，使用第 35 轮最佳断点迁移至 rec6，随后恢复训练。第 35 轮断点的验证集 Hit@5 为 0.0265617315、Hit@10 为 0.0410052314。

rec6 恢复实验最终输出的测试指标：

| 指标 | 原始数值 | 便于阅读的表示 |
|---|---:|---:|
| Loss | 8.9431467056 | — |
| Hit@5 | 0.0246389117 | 2.46% |
| Hit@10 | 0.0394401476 | 3.94% |
| NDCG@5 | 0.0155868968 | 0.01559 |
| NDCG@10 | 0.0203292742 | 0.02033 |

这些是这次运行的观测结果，不是所有新机器上的精确预期值。它是单个随机种子的基线记录，还包含跨硬件恢复，不能视为多次独立重复实验的均值，也不能据此认定模型已达到最优。

对照实验应固定数据划分、SID 和评估协议；比较模型时复用同一份 SID，比较 SID 时尽量固定推荐模型和训练策略。正式结果需要多个随机种子，并报告均值和波动。

## 13. 常见问题

| 现象 | 优先检查 |
|---|---|
| `args.eval_batch` 不存在 | 尚未应用推荐入口参数修复 |
| `from_tensors` 是未知参数 | 尚未应用数据加载接口修复 |
| `item_seqs referenced before assignment` | 样本长度计算仍位于首次赋值同一行 |
| 缓存目录不存在 | 创建 `outputs/_dataset_cache`，或应用修复脚本 |
| SID 训练完找不到 `.sem_ids` | 保存 JSON 的代码仍被注释 |
| 读取已有 SID 报返回变量未定义 | 跳过 SID 训练，直接将文件传给推荐入口；本指南未修复这个分支 |
| Hugging Face SSL 连接失败 | 检查网络或改用镜像下载，本地文件齐全后离线运行 |
| `CUDA error: invalid device ordinal` | 物理 GPU 编号或程序内可见编号不匹配 |
| 显存仍占用但利用率为 0 | 检查是否按了 `Ctrl+Z`，在原 shell 用 `jobs -l` 查看 |
| 输出 `(END)`，无法继续输入命令 | 进入了分页器，按 `q` 退出 |
| 恢复时配置不一致 | 对照断点 `configs` 和原训练命令，不要靠删除检查绕过 |
| 没有 `[RESUME]`，却从 epoch 1 开始 | 未传入环境变量，或恢复补丁未生效 |
| 训练进度正常但最终效果低 | 先确认划分和指标，再分析 SID 质量、训练收敛和对照设置 |

## 14. 用 Git 保存实验代码

代码、配置、README 和修复脚本适合提交到 Git。仓库根目录的 `.gitignore` 已排除数据、预训练模型、实验输出、日志、Conda 环境归档和备份文件；这些文件需要另外保存，不会随 `git push` 自动上传。

在实际进行实验的服务器仓库中，先检查：

```bash
cd /home/bywu/nanoGR
git status --short --branch
git diff --stat
git remote -v
```

本地仓库与服务器仓库是两份独立目录。服务器上修复过的代码，只有在该仓库中提交，或同步到另一份仓库后提交，才会成为对应的 Git 版本。修复脚本本身的提交不等于已经把修复后的源码提交。

确认忽略规则生效、修改内容正确后，只暂存明确需要的文件。例如这次实验代码包含以下文件；本地尚未应用修复时，未修改的源码文件不会产生提交差异：

```bash
git add .gitignore README.md docs/README.upstream.md prepare_nanogr.py
git add train_sid.py custom_finetune.py nanogr/tokenization/rq_vae.py nanogr/dist_dataloader.py nanogr/trainer.py
git diff --cached --stat
git diff --cached
```

查看差异进入分页器时按 `q` 退出。确认暂存区不含数据和模型后，再创建本地提交：

```bash
git commit -m "Document Beauty experiments and track training fixes"
```

提交不等于已经推送到 GitHub。原始 clone 的 `origin` 指向作者仓库时，通常没有推送权限；先在自己的 GitHub 下 Fork 或准备自己的仓库，再根据实际远程地址安排推送。不要把占位地址直接用于修改远程，也不要用强制推送解决同步问题。

之后建议每组实验记录代码提交号、运行命令、随机种子、SID 文件版本和验证/测试指标；数据和权重另行归档。

## 15. 相关资料

- [nanoGR 原仓库](https://github.com/Zhhuizhe/nanoGR)
- [本文对应的源码提交](https://github.com/Zhhuizhe/nanoGR/tree/6274cb47ce8b9e466bb17104561d1cef9f2a787c)
- [Amazon Reviews 2014 官方数据说明](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- [sentence-t5-base 模型](https://huggingface.co/sentence-transformers/sentence-t5-base)
- [PyTorch 历史版本安装命令](https://pytorch.org/get-started/previous-versions/)
- [conda-pack 官方迁移说明](https://conda.github.io/conda-pack/)
