import json
import logging
from pathlib import Path
from typing import Tuple, Any

import click
import torch
from accelerate import Accelerator
from datasets import Metric
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification

from crowd_calibrator.train import get_dataloader
from crowd_calibrator.utils import load_config, get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


def evaluate_data(
        model: Any,
        dataloader: DataLoader,
        device: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Function for running model on evaluation or test set.
    In this function, the model loaded from checkpoint is applied on the evaluation or test set from a dataloader.
    Loss and accuracy are tracked as well.
    Args:
        model (Model): Model that is being trained.:
        dataloader (DataLoader): Object that will load the training data.
        metric (Metric): Metric that is being tracked.
        device (str): Device on which training will be done.
    Returns:
        eval_loss (float): Average loss over the whole validation set.
        eval_accuracy (float): Average accuracy over the whole validation set.
    """
    model.eval()

    predictions = torch.tensor([])
    soft_references = torch.tensor([])
    hard_references = torch.tensor([])
    logits = torch.tensor([])
    internal_representation = torch.tensor([])

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            soft_references = torch.cat([soft_references, batch["labels"].to("cpu")])
            logits = torch.cat([logits, outputs.logits.to("cpu")])
            hard_references = torch.cat([hard_references, batch["hard_label"].to("cpu")])
            # according to https://github.com/huggingface/transformers/issues/1328
            internal_representation = torch.cat([internal_representation, outputs["hidden_states"][24][:, 0, :].to("cpu")])

    correct = torch.eq(predictions, hard_references.float()).int()

    return logits, soft_references, hard_references, predictions, correct, internal_representation


@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
def main(config_path: str):
    """Function that executes the entire training pipeline.
    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.
    Args:
        config_path (str): path to the config file for the training experiment.
    """
    config = load_config(config_path)
    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    save_dict = {}

    accelerator = Accelerator()
    device = accelerator.device

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    dataset_directory = config["task"].get("dataset_directory")
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device", device)
    padding = config["processing"]["padding"]

    # Load dataset and dataloaders.
    ds, tokenizer = get_dataset(
        dataset_name,
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
        dataset_directory=dataset_directory,
    )
    for split in ["train", "validation"]:
        dataset = ds[split]
        batch_size = config["pipeline"]["batch_size"]
        dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)

        # Load model.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config["task"]["num_labels"],
            output_hidden_states=True
        )
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model"])

        model, dataloader = accelerator.prepare(
            model, dataloader
        )

        logger.info(f" Device: {device}.")
        logger.info(" Starting evaluating model on the data.")
        logits, soft_references, hard_references, predictions, correct, internal_representation = evaluate_data(
            model,
            dataloader,
            device
        )

        save_dict[split] = {
            "predictions": predictions.tolist(),
            "logits": logits.tolist(),
            "soft_references": soft_references.tolist(),
            "hard_references": hard_references.tolist(),
            "correct": correct.tolist(),
            "internal_representation": internal_representation.tolist(),
        }

    if "output_predictions" in config["pipeline"]:
        p = Path(config["pipeline"]["output_predictions"]).parent
        p.mkdir(exist_ok=True, parents=True)

        with open(config["pipeline"]["output_predictions"], "w") as f:
            json.dump(save_dict, f)


if __name__ == "__main__":
    main()
