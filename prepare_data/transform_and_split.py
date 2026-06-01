import json
import random
import os
import argparse


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"Saved {len(data)} examples to {path}")


def split_data(processed_data, seed):
    print("Splitting data (80/10/10)...")
    random.seed(seed)
    random.shuffle(processed_data)

    n = len(processed_data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = processed_data[:train_end]
    val_data = processed_data[train_end:val_end]
    test_data = processed_data[val_end:]

    return train_data, val_data, test_data


def transform_data(data, contexts):
    processed_data = []
    print("Transforming data...")
    for item in data:
        # Remove whitespaces
        instruction = item.get('instruction', '').strip()
        user_input = item.get('input', '').strip()
        output = item.get('output', '').strip()
        context_id = item.get('context_id')

        system_prompt = instruction

        # Include context if available, differentiating clearly between context and question
        if context_id and context_id in contexts and contexts[context_id].strip():
            user_prompt = f"## Kontekst:\n{contexts[context_id].strip()}\n\n## Vprašanje:\n{user_input}"
        else:
            user_prompt = user_input

        # Convert the data to prompt/completion format using standard messages format
        prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        completion = [
            {
                "role": "assistant",
                "content": output
            }
        ]

        processed_data.append({
            "id": item.get('id'),
            "context_id": context_id,
            "type": item.get('type'),
            "prompt": prompt,
            "completion": completion
        })

    return processed_data


def main():
    parser = argparse.ArgumentParser(description="Transform and split GaMS-Instruct-MED_2.0 dataset.")
    parser.add_argument("--data_file", type=str, default="../data/GaMS-Instruct-MED_2.0/GaMS-Instruct-MED_2.0.json")
    parser.add_argument("--context_file", type=str,
                        default="../data/GaMS-Instruct-MED_2.0/GaMS-Instruct-MED_2.0_context.json")
    parser.add_argument("--out_dir", type=str, default="../data/GaMS-Instruct-MED_2.0_processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load contexts
    contexts = {}
    print(f"Loading contexts from {args.context_file}...")
    with open(args.context_file, 'r', encoding='utf-8') as f:
        ctx_list = json.load(f)
        for c in ctx_list:
            if 'context_id' in c and 'context' in c:
                contexts[c['context_id']] = c['context']

    # Load data
    print(f"Loading data from {args.data_file}...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = transform_data(data, contexts)

    train_data, val_data, test_data = split_data(processed_data, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")

    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    save_jsonl(test_data, test_path)

    print(f"Processed total {len(processed_data)} examples.")
    print(f"Train: {len(train_data)} examples -> {train_path}")
    print(f"Val:   {len(val_data)} examples -> {val_path}")
    print(f"Test:  {len(test_data)} examples -> {test_path}")


if __name__ == "__main__":
    main()
