#!/bin/bash
#SBATCH --job-name=merge_lora
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/dev/null

# TODO: Add path to the root dir of the GaMS-PEFT-Tuning repository
WORK_DIR=

CONTAINER_DIR=$WORK_DIR/apptainer
CONTAINER=$CONTAINER_DIR/slaif_peft.sif
MODELS_DIR=$WORK_DIR/models

SCRIPT_DIR=$WORK_DIR/merge_lora
ADAPTERS_DIR=$WORK_DIR/training/experiments

base_model_path=$1
tokenizer_path=$2
adapter_checkpoint=$3
output_name=$4

ADAPTER_PATH=$ADAPTERS_DIR/$adapter_checkpoint

mkdir -p $SCRIPT_DIR/logs

# run with singularity
srun \
  --cpu-bind=verbose \
  --output=$SCRIPT_DIR/logs/"${output_name}.txt" \
  singularity exec \
    --nv \
    -B $MODELS_DIR:/models,$ADAPTER_PATH:/adapter_input_path,$SCRIPT_DIR:/script \
    $CONTAINER \
    python3 /script/merge.py \
    --base_model=$base_model_path \
    --tokenizer=$tokenizer_path \
    --adapter=/adapter_input_path \
    --merged_model=/models/$output_name
