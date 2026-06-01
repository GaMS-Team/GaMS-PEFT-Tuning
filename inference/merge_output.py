import json
import os


def load_data(input_file, data):
    with open(input_file, "r") as f_in:
        for line in f_in:
            data.append(json.loads(line))


def main(input_dir, output_file):
    data = []

    for file in os.listdir(input_dir):
        if file.endswith(".jsonl"):
            load_data(os.path.join(input_dir, file), data)

    with open(output_file, "w") as f_out:
        for example in data:
            f_out.write(json.dumps(example) + "\n")

    print(f"Saved {len(data)} examples to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()
    main(args.input_dir, args.output_file)
