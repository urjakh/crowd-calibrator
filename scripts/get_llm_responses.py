import json
from collections import defaultdict
from pathlib import Path

import click
from datasets import load_from_disk, load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import os

from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_prompt(sample, task: str):
    if task == "hatespeech":
        sentence = sample["text"]
        prompt = f"Classify with one of the following labels: (a) not hate speech, (b) hate speech, " \
                 f"and the probability that it is correct (0.0 to 1.0) for the following sentence. Give " \
                 f"ONLY the label and probability, no other words or explanation. Follow this format:\n\n" \
                 f"Label: <most likely label, either a or b, as short as possible; not a complete sentence, just the " \
                 f"label!>\n Probability: <the probability between 0.0 and 1.0 that your guess is correct, " \
                 f'without any extra commentary whatsoever; just the probability!>\n\nThe sentence is: """{sentence}""".'
    else:
        prompt = f""

    return prompt


def get_llm_response(prompt, model, seed=0):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt}
        ],
        seed=seed,
        logprobs=True,
        max_tokens=50,
        top_logprobs=5,
    )
    return completion


@click.command()
@click.option("-d", "--dataset", "dataset", required=True, type=str)
@click.option("-s", "--split", "split", required=True, type=str)
@click.option("-o", "--output-path", "output_path", required=True, type=str)
@click.option("-m", "--model", "model", type=str, default="gpt-4-1106-preview")
@click.option("-t", "--task", "task", type=str, default="hatespeech")
def main(dataset: str, split: str, output_path: str, model: str = "gpt-4-1106-preview", task="hatespeech"):
    if Path(dataset).is_dir():
        dataset = load_from_disk(dataset)[split]
    else:
        dataset = load_dataset(dataset)[split]

    comment_id_to_results = defaultdict(dict)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        prompt = get_prompt(sample, task)
        result = get_llm_response(prompt, model)

        logprobs = result.choices[0].logprobs.content

        comment_id_to_results[sample["comment_id"]] = {
            "input": prompt,
            "output": result.choices[0].message.content,
            "logprobs": {logprobs[j].token: logprobs[j].logprob for j in range(len(logprobs))},
            "top_logprobs": [{option.token: option.logprob for option in token.top_logprobs} for token in logprobs],
        }

        if i % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(comment_id_to_results, f)

    with open(output_path, "w") as f:
        json.dump(comment_id_to_results, f)


if __name__ == "__main__":
    main()
