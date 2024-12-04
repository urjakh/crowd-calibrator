# Code modified from:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py
# Changed structure of the file, removed unnecessary code (e.g. creating new functions and removing irrelevant code),
# added comments, and added other necessary code for this research.
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import logging
import math
import random
from collections import defaultdict
from typing import Callable, Tuple, Any

import click
import evaluate
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from evaluate import Metric
from scipy.spatial.distance import jensenshannon
from scipy import stats
from torch.nn.functional import log_softmax, kl_div, softmax
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification
from transformers import set_seed, default_data_collator, DataCollatorWithPadding, get_scheduler

from crowd_calibrator.utils import get_dataset, load_config, save_model, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")

from warnings import filterwarnings
filterwarnings("ignore")


class BestEpoch:
    """Class to keep track of the best epoch while training.

    Keeps track by comparing the evaluation loss of each epoch. If it is lower than the current best loss, this epoch
    will be considered the best until now, and its evaluation loss will replace the previous best loss.

    Attributes:
        best_epoch (int): Integer indicating the best epoch until now.
        best_loss (float): Float indicating the best loss until now.
        best_score (float): Float indicating the best score until now.

    """
    def __init__(self):
        """Initialize the tracker of the best epoch."""
        self.best_epoch: int = 0
        self.best_loss: float = float("inf")
        self.best_score: float = 0.0

    def update(self, current_loss: float, current_score: float, epoch: int) -> None:
        """Updates the best epoch tracker.

        Takes the evaluation loss and score of the current epoch and compares it with the current best loss.
        If it is lower, updates the current loss, score, and epoch to be the best until now.

        Args:
            current_loss (float): loss of the current epoch.
            current_score (float): score of the current epoch.
            epoch (int): which epoch.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_score = current_score
            self.best_epoch = epoch


def get_optimizer(model: Any, learning_rate: float, weight_decay: float) -> Optimizer:
    """Function that returns the optimizer for training.

    Given the model, learning rate, and weight decay, this function returns the optimizer that can be used while
    training. The model parameters are split into two groups: weight decay and non-weight decay groups, as done in the
    BERT paper.

    Args:
        model (torch.nn.module): Model used for training.
        learning_rate (float): Float that indicates the learning rate.
        weight_decay (float): Float that indicates the weight decay.

    Returns:
        optimizer (Optimizer): optimizer for the training.
    """
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer


def get_dataloader(
        dataset: Dataset, tokenizer: Callable, batch_size: int, padded: bool = False, shuffle: bool = False
) -> DataLoader:
    """Function that returns a dataloader.

    Given a dataset, tokenizer, batch size, and if padding has been applied already or not, a dataloader is returned
    with the appropriate data collator.

    Args:
        dataset (Dataset): Dataset that will be loaded.
        tokenizer (Tokenizer): Tokenizer that will be used if padding has not been applied before.
        batch_size (int): Batch size of the loader.
        padded (bool): Boolean that implies if the data has already been padded or not.
        shuffle (bool): Boolean to indicate if data should be shuffled by dataloader.

    Returns:
        dataloader (DataLoader): Dataloader that loads the dataset.
    """
    # If dataset has been padded already, use default data collator. Else, use collator with padding.
    if padded:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size, drop_last=True)
    return dataloader


def get_loss(outputs, target, loss_type="hard"):
    """
        Computes the loss based on the loss type being used.
    """
    output = outputs.logits
    if loss_type == "cross_entropy":
        loss = torch.nn.functional.cross_entropy(output, target)
    elif loss_type == "kl":
        output = log_softmax(output, dim=1)
        loss = torch.nn.functional.kl_div(output, target, reduction="batchmean")
    elif loss_type == "mse":
        output = softmax(output, dim=1)
        loss = torch.nn.functional.mse_loss(output, target)
    elif loss_type == "jsd":
        output = softmax(output, dim=1)
        kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        output, target = output.view(-1, output.size(-1)), target.view(-1, target.size(-1))
        m = (0.5 * (output + target)).log()
        loss = 0.5 * (kl(m, output.log()) + kl(m, target.log()))
    else:
        loss = outputs.loss
    return loss


def combine_compute(self, predictions=None, references=None, **kwargs):
    """Compute each evaluation module.

    Usage of positional arguments is not allowed to prevent mistakes.

    Args:
        predictions (list/array/tensor, optional): Predictions.
        references (list/array/tensor, optional): References.
        **kwargs (optional): Keyword arguments that will be forwarded to the evaluation module :meth:`_compute`
            method (see details in the docstring).

    Return:
        dict or None

        - Dictionary with the results if this evaluation module is run on the main process (``process_id == 0``).
        - None if the evaluation module is not run on the main process (``process_id != 0``).
    """
    results = []

    for evaluation_module in self.evaluation_modules:
        if evaluation_module.name == "rocauc":
            if "average" in kwargs and kwargs["average"] == "macro":
                batch = {"prediction_scores": predictions, "references": references}
            else:
                continue
        elif evaluation_module.name == "precision" or evaluation_module.name == "recall":
            batch = {"predictions": predictions, "references": references, "zero_division": 0, **kwargs}
        elif evaluation_module.name == "accuracy":
            batch = {"predictions": predictions, "references": references}
        else:
            batch = {"predictions": predictions, "references": references, **kwargs}
        results.append(evaluation_module.compute(**batch))

    return self._merge_results(results)


def entropy_correlation(soft_labels_human, logits_model, num_labels):
    """
        Calulate the pearson correlation between the entropy of the soft labels and the entropy of the model logits.
    """
    maximum_entropy = math.log(num_labels)
    ne_human = stats.entropy(soft_labels_human, axis=1) / maximum_entropy
    soft_labels_model = softmax(logits_model)
    ne_model = stats.entropy(soft_labels_model, axis=1) / maximum_entropy
    pearson_correlation, p_value = stats.pearsonr(ne_human, ne_model)
    return pearson_correlation, p_value


def train(
        model: Any,
        epoch: int,
        dataloader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        metric: Metric,
        logging_freq: int,
        max_steps: int,
        accelerator: Accelerator,
        loss_type: str = "hard",
) -> None:
    """Function that performs all the steps during the training phase.

    In this function, the entire training phase of an epoch is run. Looping over the dataloader, each batch is fed
    to the model, the loss and metric are tracked/calculated, and the forward and backward pass are done.

    Args:
        model (Model): Model that is being trained.
        epoch (int): Current epoch of experiment.
        dataloader (DataLoader): Object that will load the training data.
        optimizer (Optimizer): Optimizer for training.
        lr_scheduler (_LRScheduler): Learning rate scheduler for the optimizer.
        metric (Metric): Metric that is being tracked.
        logging_freq (int): Frequency of logging the training metrics.
        max_steps (int): Maximum amount of steps to be taken during this epoch.
        device (str): Device on which training will be done.
    """
    model.train()
    logging.info(f" Start training epoch {epoch}")

    scores = defaultdict(list)
    losses = []
    for step, batch in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            if step < 5:
                print(batch["input_ids"][0])
                print(dataloader.collate_fn.tokenizer.batch_decode(batch["input_ids"])[0])

            labels = batch["labels"]
            if loss_type == "hard":
                labels = batch["hard_label"]

            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
            loss = get_loss(outputs, batch["labels"], loss_type=loss_type)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]

        predictions = outputs.logits.argmax(dim=-1)
        output_logits = outputs.logits
        num_labels = model.num_labels

        cross_entropy = torch.nn.functional.cross_entropy(output_logits, batch["labels"]).cpu().detach().numpy().item()
        pearson_correlation, p_value = entropy_correlation(batch["labels"].cpu(), output_logits.cpu().detach(), num_labels)
        kl_divergence = kl_div(log_softmax(output_logits), batch["labels"], reduction="batchmean").cpu().detach().numpy().item()
        jsd = np.mean(jensenshannon(batch["labels"].cpu(), softmax(output_logits.cpu().detach()), axis=1))

        score_micro = metric.compute(predictions=predictions, references=batch["hard_label"], average="micro")
        score_macro = metric.compute(predictions=predictions, references=batch["hard_label"], average="macro")
        metrics_micro = {f"train_micro_{name}": val for name, val in score_micro.items()}
        metrics_macro = {f"train_macro_{name}": val for name, val in score_macro.items()}
        metrics = metrics_micro | metrics_macro
        metrics["entropy_correlation"] = pearson_correlation
        metrics["p_value"] = p_value
        metrics["cross_entropy"] = cross_entropy
        metrics["kl_divergence"] = kl_divergence
        metrics["jsd"] = jsd
        for name, val in metrics.items():
            scores[name].append(val)

        current_step = (epoch * len(dataloader)) + step
        log_dict = {"epoch": epoch, "train_loss": loss, **metrics, "learning_rate": current_lr}
        wandb.log(log_dict, step=current_step)

        losses.append(loss.detach().cpu().numpy())

        if step % logging_freq == 0:
            logger.info(f" Epoch {epoch}, Step {step}: Loss: {loss}, Score: {metrics}")

        if current_step == max_steps - 1:
            break

    average_loss = np.mean(losses)
    metrics = {f"average_{name}": np.nanmean(scores[name]) for name in scores}

    logger.info(f" Epoch {epoch} average training loss: {average_loss}, {metrics}")

    wandb.log({"average_train_loss": average_loss, **metrics})


def validate(
        model: Any,
        epoch: int,
        dataloader: DataLoader,
        metric: Metric,
        max_steps: int,
        loss_type: str = "hard",
) -> Tuple[np.float_, float]:
    """Function that performs all the steps during the validation/evaluation phase.

    In this function, the entire evaluation phase of an epoch is run. Looping over the dataloader, each batch is fed
    to the model and the loss and score are tracked.

    Args:
        model (Model): Model that is being trained.:
        epoch (int): Current epoch of experiment.
        dataloader (DataLoader): Object that will load the training data.
        metric (Metric): Metric that is being tracked.
        max_steps (int): Maximum amount of steps to be taken during this epoch.
        device (str): Device on which training will be done.

    Returns:
        eval_loss (float): Average loss over the whole validation set.
        eval_score (float): Average score over the whole validation set.
    """
    model.eval()

    predictions = torch.tensor([])
    hard_references = torch.tensor([])
    logits = torch.tensor([])
    soft_references = torch.tensor([])

    with torch.no_grad():
        logger.info(" Starting Evaluation")
        losses = []
        for step, batch in enumerate(tqdm(dataloader)):

            if step < 5:
                print(batch["input_ids"][0])
                print(dataloader.collate_fn.tokenizer.batch_decode(batch["input_ids"])[0])

            labels = batch["labels"]
            if loss_type == "hard":
                labels = batch["hard_label"]

            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            hard_references = torch.cat([hard_references, batch["hard_label"].to("cpu")])
            logits = torch.cat([logits, outputs.logits.to("cpu")])
            soft_references = torch.cat([soft_references, batch["labels"].to("cpu")])

            loss = get_loss(outputs, batch["labels"], loss_type=loss_type)

            losses.append(loss.detach().cpu().numpy())
            current_step = (epoch * len(dataloader)) + step

            if current_step == max_steps - 1:
                break

    num_labels = model.num_labels

    cross_entropy = torch.nn.functional.cross_entropy(logits, soft_references).cpu().detach().numpy().item()
    pearson_correlation, p_value = entropy_correlation(soft_references, logits.detach(), num_labels)
    kl_divergence = kl_div(log_softmax(logits), soft_references, reduction="batchmean").cpu().detach().numpy().item()
    jsd = np.mean(jensenshannon(soft_references, softmax(logits.detach()), axis=1))

    eval_loss = np.mean(losses)
    score_micro = metric.compute(predictions=predictions, references=hard_references, average="micro")
    score_macro = metric.compute(predictions=predictions, references=hard_references, average="macro")
    metrics_micro = {f"eval_micro_{name}": val for name, val in score_micro.items()}
    metrics_macro = {f"eval_macro_{name}": val for name, val in score_macro.items()}
    metrics = metrics_micro | metrics_macro
    metrics["eval_entropy_correlation"] = pearson_correlation
    metrics["eval_p_value"] = p_value
    metrics["eval_cross_entropy"] = cross_entropy
    metrics["eval_kl_divergence"] = kl_divergence
    metrics["eval_jsd"] = jsd

    logger.info(f" Evaluation {epoch}: Average Loss: {eval_loss}, Average metrics: {metrics}")

    wandb.log({"epoch": epoch, "eval_loss": eval_loss, **metrics})

    return eval_loss, metrics


@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
def main(config_path):
    """Function that executes the entire training pipeline.

    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.

    Args:
        config_path (str): path to the config file for the training experiment.
    """
    config = load_config(config_path)

    # Initialize Weights & Biases.
    wandb.init(config=config, project=config["wandb"]["project_name"], name=config["wandb"]["run_name"])

    # Set seeds for reproducibility.
    set_seed(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    # Set Accelerator for device and speedups.
    # Set PYTORCH_ENABLE_MPS_FALLBACK=1 to make this work on Mac
    gradient_accumulation_steps = config["optimizer"].get("gradient_accumulation_steps", 1)
    mixed_precision = config["pipeline"].get("mixed_precision")
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision)
    device = accelerator.device

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    if "device" in config["pipeline"]:
        device = config["pipeline"]["device"]
    dataset_directory = config["task"].get("dataset_directory")
    padding = config["processing"]["padding"]
    loss_type = config["optimizer"]["loss_type"]

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        dataset_name,
        model_name,
        padding=padding,
        tokenize=True,
        batched=True,
        return_tokenizer=True,
        dataset_directory=dataset_directory,
    )
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    train_batch_size = config["pipeline"]["train_batch_size"]
    validation_batch_size = config["pipeline"]["validation_batch_size"]
    train_dataloader = get_dataloader(train_dataset, tokenizer, train_batch_size, padding, shuffle=True)
    validation_dataloader = get_dataloader(validation_dataset, tokenizer, validation_batch_size, padding)

    # Set amount of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    n_epochs = config["pipeline"]["n_epochs"]
    max_train_steps = n_epochs * num_update_steps_per_epoch
    # If a maximum amount of steps is specified, change the amount of epochs accordingly.
    if "max_train_steps" in config["pipeline"]:
        max_train_steps = config["pipeline"]["max_train_steps"]
        n_epochs = int(np.ceil(max_train_steps / num_update_steps_per_epoch))

    # Load metric, model, optimizer, and learning rate scheduler.
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    # Since evaluate.compute cannot handle extra arguments (e.g. average), override with own function that allows.
    metric.compute = functools.partial(combine_compute, metric)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    optimizer = get_optimizer(model, config["optimizer"]["learning_rate"], config["optimizer"]["weight_decay"])

    lr_scheduler = get_scheduler(
        name=config["optimizer"]["learning_rate_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["optimizer"]["num_warmup_steps"],
        num_training_steps=max_train_steps,
    )

    # Set everything correctly according to resumption of training or not.
    start_epoch = 0
    if "resume" in config["pipeline"]:
        model, optimizer, lr_scheduler, epoch = load_model(config["pipeline"]["resume"], model, optimizer, lr_scheduler)
        # Start from the next epoch.
        start_epoch = epoch + 1

    model = model.to(device)
    wandb.watch(model, optimizer, log="all", log_freq=10)

    print("\n")
    logger.info(f" Amount training examples: {len(train_dataset)}")
    logger.info(f" Amount validation examples: {len(validation_dataset)}")
    logger.info(f" Amount of epochs: {n_epochs}")
    logger.info(f" Amount optimization steps: {max_train_steps}")
    logger.info(f" Batch size train: {train_batch_size}, validation: {validation_batch_size}")
    logger.info(f" Device: {device}")
    logger.info(f" Mixed Precision: {mixed_precision}")
    logger.info(f" Gradient Acccumulation Steps: {gradient_accumulation_steps}")
    print("\n")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f" Sample {index} of the training set: {train_dataset[index]}.")
    print("\n")

    # Setup best epoch tracker and early stopper if present in config.
    logging_freq = config["pipeline"]["logging_freq"]
    tracker = BestEpoch()

    model, optimizer, train_dataloader, lr_scheduler, validation_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, validation_dataloader
    )

    for epoch in range(start_epoch, n_epochs):
        train(
            model,
            epoch,
            train_dataloader,
            optimizer,
            lr_scheduler,
            metric,
            logging_freq,
            max_train_steps,
            accelerator,
            loss_type=loss_type,
        )
        eval_loss, eval_score = validate(
            model,
            epoch,
            validation_dataloader,
            metric,
            max_train_steps,
            loss_type=loss_type,
        )

        print("\n")

        save_model(
            model, optimizer, lr_scheduler, epoch, config["pipeline"]["output_directory"], model_name
        )
        tracker.update(eval_loss, eval_score, epoch)

    logger.info(
        f"Best performance was during epoch {tracker.best_epoch}, with a loss of {tracker.best_loss}, "
        f"and score of {tracker.best_score}"
    )


if __name__ == "__main__":
    main()
