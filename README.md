# GaMS-PEFT-Tuning

This repository contains the code for the `Adaptation of a Large Language Model for a Specific Task using Parameter Efficient Fine-Tuning` SLAIF service. It contains an example code for GaMS3-12B SFT using LoRA and efficient batch inference using vLLM. The code and environment is optimized for Slovene HPC Vega.

---

## 1. Environment Setup

### Clone the Repository

```bash
git clone https://github.com/GaMS-Team/GaMS-PEFT-Tuning.git
cd GaMS-PEFT-Tuning
```

### What is Apptainer?

Apptainer (formerly Singularity) is a container runtime designed for HPC environments. Unlike Docker, it does not require root privileges to run, which makes it suitable for shared SLURM clusters where users do not have administrator access. It lets you package a full software environment — Python, CUDA libraries, pip packages — into a single `.sif` image file that runs reproducibly on any compatible node.

The container in this project is based on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and includes PyTorch 2.9.1, Transformers 4.56.1, TRL 0.25.1, DeepSpeed 0.18.1, PEFT, vLLM 0.16.0, FlashAttention 2, the Liger kernel, and WandB.

### Build the Apptainer Image

The recipe is located at `apptainer/recipe.def`. Build the image and store it in the same directory:

```bash
cd apptainer/
apptainer build slaif_peft.sif recipe.def
cd ..
```

Building the image requires Apptainer to be installed on the machine and may take 10–20 minutes depending on network speed and the cluster's hardware. The resulting file `apptainer/slaif_peft.sif` will be used by all SLURM job scripts. You only need to build it once.

> Note: Building a container typically requires either root access or the `--fakeroot` flag on clusters that support it. Check with your cluster administrators if the plain `apptainer build` command fails.

---

## 2. Data Preparation

### Dataset: GaMS-Instruct-MED 2.0

This pipeline uses the GaMS-Instruct-MED 2.0 dataset, a pre-prepared medical domain instruction dataset. It consists of two files:

- **`GaMS-Instruct-MED_2.0.json`** — A JSON array of instruction/response pairs. Each entry has fields: `id`, `instruction`, `input`, `output`, `type`, and optionally `context_id`.
- **`GaMS-Instruct-MED_2.0_context.json`** — A JSON array of context passages identified by `context_id`. Some examples use these passages in a RAG-style (Retrieval-Augmented Generation) fashion, where the context is prepended to the user question.

### Download and Extract the Data

Create a `data/` directory in the repository root and download the dataset there:

```bash
mkdir data/
cd data/
curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2045{/GaMS-Instruct-MED_2.0.zip}
unzip GaMS-Instruct-MED_2.0.zip
cd ..
```

After extraction, your data directory should contain:

```
data/
└── GaMS-Instruct-MED_2.0/
    ├── GaMS-Instruct-MED_2.0.json
    └── GaMS-Instruct-MED_2.0_context.json
```

### Transform and Split the Data

The script `prepare_data/transform_and_split.py` converts the raw dataset into the format expected by the training script and splits it into train/validation/test sets.

**What the script does:**

1. Loads the context file and builds a lookup dictionary by `context_id`.
2. Loads the main instruction data.
3. For each example, constructs a `prompt` (a list of chat messages with `system` and `user` roles) and a `completion` (a list with the `assistant` message). If a `context_id` is present and the corresponding context is non-empty, the context is prepended to the user question under a `## Kontekst:` heading, followed by the question under `## Vprašanje:`.
4. Shuffles the data with a fixed random seed and splits it 80% train / 10% validation / 10% test.
5. Saves three `.jsonl` files to the output directory.

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_file` | `../data/GaMS-Instruct-MED_2.0/GaMS-Instruct-MED_2.0.json` | Path to the main dataset JSON |
| `--context_file` | `../data/GaMS-Instruct-MED_2.0/GaMS-Instruct-MED_2.0_context.json` | Path to the context JSON |
| `--out_dir` | `../data/GaMS-Instruct-MED_2.0_processed` | Directory where output files are saved |
| `--seed` | `42` | Random seed for reproducible shuffling |

**Run the script:**

```bash
cd prepare_data/
python3 transform_and_split.py
cd ..
```

Or with explicit paths:

```bash
python3 prepare_data/transform_and_split.py \
    --data_file data/GaMS-Instruct-MED_2.0/GaMS-Instruct-MED_2.0.json \
    --context_file data/GaMS-Instruct-MED_2.0/GaMS-Instruct-MED_2.0_context.json \
    --out_dir data/GaMS-Instruct-MED_2.0_processed
```

This produces `train.jsonl`, `val.jsonl`, and `test.jsonl` in the output directory.

### Output Format

The pipeline uses the standard SFT (Supervised Fine-Tuning) prompt/completion format. In this format, the input to the model (the `prompt`) is kept separate from the expected output (the `completion`). This separation tells the trainer to compute the loss only on the completion tokens, not on the prompt — which is the correct behaviour for instruction fine-tuning.

Each line in the output JSONL files looks like this:

```json
{
  "id": "example_001",
  "context_id": "ctx_042",
  "type": "qa",
  "prompt": [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "## Kontekst:\n<context text>\n\n## Vprašanje:\n<question text>"}
  ],
  "completion": [
    {"role": "assistant", "content": "<expected answer>"}
  ]
}
```

For examples without a context, the `user` content is just the question text with no heading.

---

## 3. LoRA Training

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all weights of a large model, LoRA inserts small trainable matrices (adapters) into specific layers and freezes the original model weights. This dramatically reduces the number of trainable parameters and GPU memory required, while achieving fine-tuning quality close to full fine-tuning.

### Code Location

All training code is in `lora_training/`:

```
lora_training/
├── gams_sft.py           # Main training script
├── deepspeed_config.json # DeepSpeed configuration
└── run_sft.sbatch        # SLURM job submission script
```

### Before Running: Required Setup

Open `lora_training/run_sft.sbatch` and fill in the two required values:

```bash
# TODO: Add path to the root dir of the GaMS-PEFT-Tuning repository
WORK_DIR=/path/to/GaMS-PEFT-Tuning

WANDB_API_KEY=your_wandb_api_key_here
```

See the WandB section below for details on obtaining an API key.

### Submitting the Training Job

The sbatch script accepts the following CLI arguments: `--lora_rank`, `--warmup_steps`, `--learning_rate`, and `--min_lr`. Submit the job like this:

```bash
sbatch lora_training/run_sft.sbatch \
    --lora_rank 64 \
    --warmup_steps 500 \
    --learning_rate 2e-5 \
    --min_lr 1e-6
```

The job requests 4 nodes, 4 GPUs per node (16 GPUs total), 32 CPUs per task, and exclusive node access. Training runs for up to 2 days (`--time=2-00`).

### Key Training Configuration

**LoRA settings** (in `gams_sft.py`, function `use_lora`):

- `r` (rank): controlled by `--lora_rank`, suggested default 64. Higher rank means more trainable parameters and potentially better quality, at the cost of more memory and compute.
- `lora_alpha`: set to `2 * rank`. This is the scaling factor for LoRA updates.
- `lora_dropout`: 0.1. Regularization to prevent overfitting.
- `target_modules`: all 7 projection layers — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Targeting all projection modules (rather than just query and value) typically improves fine-tuning quality.

**Data and batching:**

- `MAX_LENGTH`: 16384 tokens. Examples whose prompt exceeds this length are filtered out before training.
- `micro_batch_size`: 1 per GPU.
- `batch_size`: 64 globally. The script automatically computes `gradient_accumulation_steps = 64 / (world_size * 1)`. With 16 GPUs this gives 4 gradient accumulation steps.

**Training schedule:**

- 3 epochs over the training set.
- `cosine_with_min_lr` learning rate scheduler, with warmup controlled by `--warmup_steps` and a minimum LR floor set by `--min_lr`.

### Optimization Techniques

**DeepSpeed ZeRO Stage 3** (`deepspeed_config.json`):

DeepSpeed is a library that enables training very large models across multiple GPUs and nodes. ZeRO (Zero Redundancy Optimizer) Stage 3 shards model parameters, gradients, and optimizer states across all GPUs — not just gradients and optimizer states as in earlier stages. This means even the model weights themselves are distributed, allowing models much larger than a single GPU's memory to be trained. Key settings:

- `bf16.enabled: true` — uses bfloat16 precision, which is more numerically stable than float16 for training.
- `stage3_gather_16bit_weights_on_model_save: true` — reassembles the full model weights when saving a checkpoint, so the saved checkpoint is in standard format.
- `overlap_comm: true` — overlaps gradient communication with computation for better throughput.
- Batch sizes and gradient accumulation are set to `"auto"` so DeepSpeed reads them from the training arguments at runtime.

**Flash Attention 2:**

Flash Attention is a memory-efficient and faster implementation of the transformer self-attention mechanism. The model is loaded with `attn_implementation="flash_attention_2"`, which reduces memory usage and speeds up training, especially for long sequences.

**Liger Kernel:**

Liger Kernel provides fused, memory-efficient GPU kernels for common LLM operations (e.g., RMSNorm, RoPE, cross-entropy loss). It reduces peak memory usage and improves throughput with minimal code changes.

**Gradient Checkpointing:**

Normally, all intermediate activations are kept in memory during the forward pass so gradients can be computed during the backward pass. Gradient checkpointing trades compute for memory by recomputing activations on the fly during the backward pass instead of storing them. This is enabled in the training configuration and significantly reduces memory usage at the cost of a modest slowdown (~20–30%).

### WandB (Weights & Biases) Logging

WandB is an experiment tracking platform that logs training metrics (loss, learning rate, etc.) to a web dashboard in real time. To use it:

1. Create a free account at [https://wandb.ai](https://wandb.ai).
2. Find your API key under Settings > API Keys.
3. Set `WANDB_API_KEY` in the sbatch script to your key.

If you do not want to use WandB, remove or comment out the WandB-related lines in `gams_sft.py` (where `report_to="wandb"` is set in `SFTConfig`) and set `report_to="none"` instead.

### Resuming Training from a Checkpoint

If a job is interrupted, you can resume from the last saved checkpoint. Checkpoints are stored inside the experiment directory under `training/experiments/<run_name>/`. To resume from a specific checkpoint, you should uncomment the following line in the `run_sft.sbatch` script and set the checkpoint number:

```bash
# TODO: Uncomment and correct the checkpoint number if you are resuming from a checkpoint
# CKPT_PATH=$OUTPUT_DIR/checkpoint-42
```

When this line is commented, the training starts from scratch.

---

## 4. Merge LoRA Adapter

### Code Location

```
merge_lora/
├── merge.py          # Merging script
└── run_merge.sbatch  # SLURM job submission script
```

### How LoRA Merging Works

During training, LoRA keeps the base model weights frozen and only trains the small adapter matrices. At inference time, the adapter weights can either be loaded separately (applied on top of the frozen base model) or permanently merged into the base model weights. Merging adds the adapter contribution directly into the base model parameters and then discards the separate adapter structure, producing a standard model checkpoint with no additional dependencies.

### When to Merge (and When Not To)

Merging is **not required** for inference. One advantage of keeping adapters separate is that a single base model can serve as the foundation for multiple fine-tuned variants — each adapter is small (tens to hundreds of MB depending on rank), so you can maintain several task-specific or version-specific adapters without duplicating the multi-GB base model. Merging is the right choice when you want a self-contained, portable checkpoint ready for deployment.

### Checkpoint Path Format

Checkpoints saved during training follow this naming convention:

```
training/experiments/<version>/checkpoint-<step_number>/
```

When calling the merge script, the adapter path should point to a specific checkpoint, for example:

```
<version>/checkpoint-<ckpt_number>
```

### Before Running: Required Setup

Open `merge_lora/run_merge.sbatch` and set `WORK_DIR`:

```bash
WORK_DIR=/path/to/GaMS-PEFT-Tuning
```

### Running the Merge Job

The sbatch script takes four positional arguments:

1. `base_model_path` — path to the base model (the starting point of the LoRA training script)
2. `tokenizer_path` — path to the tokenizer that the merged model will use (can be the same as the base model path)
3. `adapter_checkpoint` — checkpoint path relative to `training/experiments/`
4. `output_name` — name for the merged model directory under `models/`

```bash
sbatch merge_lora/run_merge.sbatch \
    cjvt/GaMS3-12B \
    cjvt/GaMS3-12B-Instruct \
    r-64_ws-500_lr-2e-5_min-lr-1e-6/checkpoint-1234 \
    GaMS3-12B-Med
```

The merged model is saved to `models/<output_name>/` and logs are written to `merge_lora/logs/<output_name>.txt`. The job runs on a single GPU and requests 64 GB of RAM with a 30-minute time limit.

The `merge.py` script loads the base model in `bfloat16`, applies the LoRA adapter using `PeftModel.from_pretrained`, calls `merge_and_unload()` to permanently fuse the adapter weights, and saves the result with `save_pretrained`. The tokenizer is also saved alongside the model so the output directory is fully self-contained.

---

## 5. vLLM Batch Inference

### Code Location

```
inference/
├── vllm_batch_inference.py  # Inference script
├── run_inference.sbatch     # SLURM array job script
├── merge_output.py          # Merge sharded outputs
└── analyze.py               # Human-readable output inspection
```

### What is vLLM?

vLLM is a high-throughput inference engine for LLMs. Its key innovation is **PagedAttention**, which manages the KV (key-value) cache using virtual memory paging. This eliminates memory fragmentation and allows batching many requests together efficiently. Compared to running inference with the standard Transformers `generate()` method, vLLM achieves significantly higher throughput, which matters when processing thousands of test examples.

### How Batch Inference Works

The script `vllm_batch_inference.py` loads the model once using vLLM, then processes the test dataset in batches using `model.chat()`. Sampling parameters are set to temperature 0.6, top-p 0.9, and a maximum of 32768 output tokens. Results are written to a JSONL file incrementally (flushed after each batch), so partial results are preserved if the job is interrupted.

### Tensor Parallelism

For large models that do not fit on a single GPU, vLLM supports tensor parallelism: the model's weight matrices are split across multiple GPUs. The `--tp_size` argument controls how many GPUs to split across. In the sbatch script, `--tp_size $SLURM_GPUS_ON_NODE` automatically uses all GPUs on the allocated node (4 in the default configuration).

### SLURM Array Jobs and Dataset Sharding

The sbatch script uses `#SBATCH --array=0-7` to launch 8 independent jobs simultaneously, each processing a different slice of the test dataset. This is dataset sharding: each job receives the full dataset but processes only every 8th example starting from its array index (`dataset[shard_idx::n_shards]`). The 8 output files are later merged into one.

`SLURM_ARRAY_TASK_ID` (0–7) is automatically set by SLURM for each array job and is used as the shard index. `SLURM_ARRAY_TASK_COUNT` (8) is used as the total number of shards. This approach scales throughput linearly with the number of array tasks, as long as enough GPU nodes are available.

### Before Running: Required Setup

Open `inference/run_inference.sbatch` and set `WORK_DIR`:

```bash
WORK_DIR=/path/to/GaMS-PEFT-Tuning
```

### Running Inference

```bash
sbatch inference/run_inference.sbatch <model_name> <batch_size>
```

- `model_name` — name of the model directory under `models/` (e.g., `GaMS3-12B-Med`)
- `batch_size` — number of examples to process concurrently (e.g., `16`)

Example:

```bash
sbatch inference/run_inference.sbatch my_run_merged 16
```

Each array task writes its output to `inference/output/<model_name>/<shard_idx>.jsonl` and logs to `inference/logs/<model_name>/<shard_idx>.log`. The output JSONL files contain the original example fields plus an added `response` field with the model's generated text.

**Key parameters in `vllm_batch_inference.py`:**

| Argument | Description |
|---|---|
| `--model_path` | Path to the model inside the container (`/models/<model_name>`) |
| `--input_file` | Path to the test JSONL file (`/data/test.jsonl`) |
| `--output_file` | Path for the output JSONL file |
| `--batch_size` | Number of examples per batch (default: 8) |
| `--tp_size` | Number of GPUs for tensor parallelism (default: 1) |
| `--n_shards` | Total number of shards (matches array job count) |
| `--shard_idx` | This job's shard index (0-based) |

---

## 6. Prediction Analysis

After all inference array jobs complete, you will have multiple sharded output files. Use the two utility scripts to combine and inspect them.

### Merging Sharded Outputs

`inference/merge_output.py` collects all `.jsonl` files from a directory and writes them into a single output file. This is the step that reassembles the sharded results from the array job.

```bash
python3 inference/merge_output.py \
    --input_dir inference/output/GaMS3-12B-Med \
    --output_file inference/output/GaMS3-12B-Med_merged.jsonl
```

- `--input_dir` — directory containing the per-shard `.jsonl` output files
- `--output_file` — path for the combined output file

The script reads every `.jsonl` file in the input directory and writes all examples into the single output file. The ordering of examples across shards is not guaranteed, so if order matters for your analysis, sort by `id` after merging.

### Inspecting Predictions

`inference/analyze.py` reads the merged JSONL file and writes the first 30 examples to a human-readable text file, showing the prompt and model response for each example with clear separators. This is useful for a quick qualitative check of model output quality.

```bash
python3 inference/analyze.py \
    --input_path inference/output/GaMS3-12B-Med_merged.jsonl \
    --output_path inference/output/GaMS3-12B-Med_sample.txt
```

- `--input_path` — path to the merged JSONL file
- `--output_path` — path for the output text file

The output text file will contain entries formatted like:

```
--- Example 1 ---
PROMPT: <system message>

<user message>

RESPONSE: <model response>

------------------------------
```

The script stops after 30 examples. To inspect more, increase the limit in the `if line_number == 30: break` line in the script.

---

## Acknowledgments

The service was developed at the **University of Ljubljana, Faculty of Computer and Information Science** as part of the Slovenian AI Factory (SLAIF) project.

The project was also supported by:

* **ARIS** (Slovenian Research and Innovation Agency).
* **NextGenerationEU**.
* European Union under Horizon Europe (101186647 – **AI4DH**)
* **EuroHPC JU**.
* **SLING** (Slovenian National Supercomputing Network).

---

## Contact

**Domen Vreš**  
domen.vres@fri.uni-lj.si

---

## License

This project is licensed under the **MIT** license.

