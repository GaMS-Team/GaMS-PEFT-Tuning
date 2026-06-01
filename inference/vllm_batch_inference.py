import json
from math import ceil
from argparse import ArgumentParser
from tqdm import trange
from vllm import LLM, SamplingParams


def generate_responses(prompts, model, sampling_params):
    responses = model.chat(prompts, sampling_params, use_tqdm=False)
    predictions = [response.outputs[0].text for response in responses]

    return predictions


def process_batch(batch, model, sampling_params):
    prompts = [example["prompt"] for example in batch]
    responses = model.chat(prompts, sampling_params, use_tqdm=False)
    predictions = [response.outputs[0].text for response in responses]

    return predictions


def process_dataset(input_file, output_file, model, sampling_params, batch_size, n_shards, shard_idx):
    with open(input_file, "r") as f_in:
        dataset = [json.loads(line) for line in f_in]

    print(f"Found {len(dataset)} examples for inference.")
    dataset = dataset[shard_idx::n_shards]
    print(f"Processing shard {shard_idx} of {n_shards} with {len(dataset)} examples.")

    f_out = open(output_file, "w")

    n_batches = ceil(len(dataset) / batch_size)

    # Using trange for progress bar
    for i in trange(n_batches, desc="Processing Batches"):
        batch = dataset[i * batch_size: min(len(dataset), (i + 1) * batch_size)]
        batch_results = process_batch(
            batch,
            model,
            sampling_params
        )

        for example, prediction in zip(batch, batch_results):
            example["response"] = prediction
            f_out.write(json.dumps(example) + "\n")
            f_out.flush()  # Ensure data is written incrementally

    f_out.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the JSONL file with outputs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of concurrent requests to process."
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs to split model across)."
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="Number of shards to split the dataset into."
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Index of the shard to process."
    )
    return parser.parse_args()


def main(args):
    print(f"Initializing vLLM model ...")

    model = LLM(
        args.model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp_size
    )

    # Dictionary for sampling params instead of vllm object
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=32768
    )

    process_dataset(
        args.input_file,
        args.output_file,
        model,
        sampling_params,
        args.batch_size,
        args.n_shards,
        args.shard_idx
    )

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
