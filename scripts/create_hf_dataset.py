import re
from collections import Counter
from pathlib import Path

import click
import emoji
import torch
from datasets import load_dataset

from torch.nn.functional import softmax

from crowd_calibrator.utils import dataset_to_input_output


class HFDatasetCreator:
    """
        TBD.
    """
    def __init__(self, dataset_name: str, dataset_file: str, seed: int = 0):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.seed = seed
        self.dataset = None

    def load_data_from_file(self):
        suffix = Path(self.dataset_file).suffix[1:]
        self.dataset = load_dataset(suffix, data_files=self.dataset_file)

    def load_data_from_name(self):
        self.dataset = load_dataset(self.dataset_file)

    def save_dataset(self, output: str):
        self.dataset.save_to_disk(output)

    def prepare_mathew(self):
        def get_labels(example):
            labels = example["annotators"]["label"]
            soft_labels = [labels.count(i) for i in range(len(labels))]
            soft_labels = softmax(torch.Tensor(soft_labels), dim=0)
            hard_label = Counter(labels).most_common(1)[0][0]
            example["soft_label"] = soft_labels
            example["hard_label"] = hard_label
            return example

        def tokens_to_sentence(example):
            example["sentence"] = " ".join(example["post_tokens"])
            return example

        dataset = self.dataset.map(get_labels)
        self.dataset = dataset.map(tokens_to_sentence)

    def clean_data(self, emojis: bool = True, urls: bool = True, usernames: bool = True):
        def remove_emojis(example):
            input_key = dataset_to_input_output[self.dataset_name]["input"]
            example[input_key] = emoji.demojize(example[input_key])
            return example

        def remove_urls(example):
            input_key = dataset_to_input_output[self.dataset_name]["input"]

            url_pattern = re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[^\s]+"
            )
            example[input_key] = re.sub(url_pattern, "[URL]", example[input_key])

            return example

        def remove_usernames(example):
            input_key = dataset_to_input_output[self.dataset_name]["input"]

            username_pattern = re.compile(r'@[A-Za-z0-9_]+')
            example[input_key] = re.sub(username_pattern, "[USER]", example[input_key])
            return example

        if emojis:
            self.dataset = self.dataset.map(remove_emojis)
        if urls:
            self.dataset = self.dataset.map(remove_urls)
        if usernames:
            self.dataset = self.dataset.map(remove_usernames)


@click.command()
@click.option("-n", "--dataset-name", "dataset_name", required=True, type=str)
@click.option("-o", "--output", "output", required=True, type=str)
@click.option("-s", "--dataset-split", "dataset_split", type=str)
def main(dataset_name: str, output: str, dataset_split: str):
    mathew_creator = HFDatasetCreator(dataset_name, "hatexplain", dataset_split)
    mathew_creator.load_data_from_name()
    mathew_creator.prepare_mathew()
    mathew_creator.clean_data()
    mathew_creator.save_dataset(output)


if __name__ == "__main__":
    main()
