import json
from argparse import ArgumentParser


def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line_number, line in enumerate(infile, 1):
            data = json.loads(line)

            prompt = "\n\n".join([turn["content"] for turn in data["prompt"]])
            response = data["response"]

            # Writing to text file with clear separators
            outfile.write(f"--- Example {line_number} ---\n")
            outfile.write(f"PROMPT: {prompt}\n\n")
            outfile.write(f"RESPONSE: {response}\n\n")
            outfile.write("-" * 30 + "\n\n")

            if line_number == 30:
                break

    print(f"Processing complete!")


# Usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output text file.")
    args = parser.parse_args()
    process_jsonl(args.input_path, args.output_path)
