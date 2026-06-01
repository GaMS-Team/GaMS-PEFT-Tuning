#!/bin/bash
#SBATCH --job-name=batch_inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=0-7
#SBATCH --output=/dev/null

# TODO: Add path to the root dir of the GaMS-PEFT-Tuning repository
WORK_DIR=

CONTAINER_DIR=$WORK_DIR/apptainer
CONTAINER=$CONTAINER_DIR/slaif_peft.sif

model_path=$1
batch_size=$2

N_SHARDS=$SLURM_ARRAY_TASK_COUNT
SHARD_IDX=$SLURM_ARRAY_TASK_ID

DATA_DIR=$WORK_DIR/data/GaMS-Instruct-MED_2.0_processed
INPUT_FILE=/data/test.jsonl

SCRIPT_DIR=$WORK_DIR/inference

OUTPUT_DIR=$SCRIPT_DIR/output/$model_path
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=/output/$SHARD_IDX.jsonl

LOG_DIR=$SCRIPT_DIR/logs/$model_path
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/$SHARD_IDX.log

srun \
    --output=$LOG_FILE \
    apptainer exec \
    --nv \
    --bind $MODELS_DIR:/models,$SCRIPT_DIR:/script,$OUTPUT_DIR:/output,$DATA_DIR:/data \
    --workdir /script \
    $CONTAINER \
    python3 vllm_batch_inference.py \
        --model_path $model_path \
        --input_file $INPUT_FILE \
        --output_file $OUTPUT_FILE \
        --batch_size $batch_size \
        --tp_size $SLURM_GPUS_ON_NODE \
        --n_shards $N_SHARDS \
        --shard_idx $SHARD_IDX
