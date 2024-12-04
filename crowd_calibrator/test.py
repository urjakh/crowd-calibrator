import functools
import json
import logging
import math
from pathlib import Path
from typing import Tuple, Any, Dict

import click
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Metric
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import confusion_matrix
from torch.nn.functional import kl_div, log_softmax, softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification

from crowd_calibrator.train import get_dataloader, combine_compute, entropy_correlation, get_loss
from crowd_calibrator.utils import load_config, get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


def jensen_shannon_divergence(logits, references):
    """
        Calculate Jensen Shannon Divergence given the model logits and the actual human label distribution.
    """
    output = softmax(logits, dim=1)
    kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    output, target = output.view(-1, output.size(-1)), references.view(-1, references.size(-1))
    m = (0.5 * (output + target)).log()
    return 0.5 * (kl(m, output.log()) + kl(m, target.log()))


def tvd(model_probs, human_probs, mean_per=None):
    """
    SOURCE: https://github.com/jsbaan/calibration-on-disagreement-data/blob/main/calibration_metrics.py.
    Computes TVD scores allowing for multiple sub-samples and groups (=classifiers).

    p: classifiers [G, 1, N, C]
    q: MLE given (sub-samples of) annotations [1, S, N, C]

    returns:
        tvd: [G, S, N] (mean_per=None), [G, S] (mean_per=sample), [G, N] (mean_per=instance)
    """
    assert model_probs.max() <= 1.0 and model_probs.min() >= 0
    assert human_probs.max() <= 1.0 and human_probs.min() >= 0

    tvds = np.sum(np.abs(model_probs - human_probs), axis=-1) / 2
    if mean_per is not None:
        if mean_per == "instance":
            tvds = tvds.mean(1)
        elif mean_per == "sample":
            tvds = tvds.mean(2)
    return tvds


def ece(true_labels, pred_labels, confidences, num_bins=10):
    """
    SOURCE: https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
    Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return ece


def evaluate_data(
        model: Any,
        dataloader: DataLoader,
        metric: Metric,
        device: str,
        loss_type: str,
) -> Tuple[np.float_, Dict, list, Any, Any, Any]:
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

    with torch.no_grad():
        losses = []
        all_predictions = []
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            predictions = torch.cat([predictions, outputs.logits.argmax(dim=-1).to("cpu")])
            soft_references = torch.cat([soft_references, batch["labels"].to("cpu")])
            logits = torch.cat([logits, outputs.logits.to("cpu")])
            hard_references = torch.cat([hard_references, batch["hard_label"].to("cpu")])

            loss = get_loss(outputs, batch["labels"], loss_type=loss_type)
            losses.append(loss.detach().cpu().numpy())
            all_predictions.extend(predictions.tolist())

    cross_entropy = torch.nn.functional.cross_entropy(logits, soft_references).item()
    pearson_correlation, p_value = entropy_correlation(soft_references, logits, model.num_labels)
    kl_divergence = kl_div(log_softmax(logits), soft_references, reduction="batchmean").item()
    jsd = np.mean(jensenshannon(soft_references, softmax(logits), axis=1))
    jsdivergence = jensen_shannon_divergence(logits, soft_references).numpy()
    total_vd = np.mean(tvd(np.array(softmax(logits)), np.array(soft_references)))
    expected_ce = ece(np.array(hard_references), np.array(predictions), np.array(softmax(logits).max(dim=-1).values))

    eval_loss = np.mean(losses)
    score_micro = metric.compute(predictions=predictions, references=hard_references, average="micro")
    score_macro = metric.compute(predictions=predictions, references=hard_references, average="macro")
    metrics_micro = {f"eval_micro_{name}": val for name, val in score_micro.items()}
    metrics_macro = {f"eval_macro_{name}": val for name, val in score_macro.items()}
    metrics = metrics_micro | metrics_macro
    metrics["entropy_correlation"] = float(pearson_correlation)
    metrics["p_value"] = float(p_value)
    metrics["cross_entropy"] = float(cross_entropy)
    metrics["kl_divergence"] = float(kl_divergence)
    metrics["jsd"] = float(jsd)
    metrics["jsdivergence"] = float(jsdivergence * math.log2(math.e))
    metrics["tvd"] = float(total_vd)
    metrics["ece"] = float(expected_ce)

    cm = confusion_matrix(hard_references, predictions)
    return eval_loss, metrics, all_predictions, cm, logits, soft_references


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

    accelerator = Accelerator()
    device = accelerator.device

    # Get values from config.
    model_name = config["task"]["model_name"]
    dataset_name = config["task"]["dataset_name"]
    dataset_directory = config["task"].get("dataset_directory")
    checkpoint_path = config["task"]["checkpoint"]
    device = config["pipeline"].get("device", device)
    padding = config["processing"]["padding"]
    loss_type = config["pipeline"]["loss_type"]

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
    dataset = dataset["test"]
    batch_size = config["pipeline"]["batch_size"]
    dataloader = get_dataloader(dataset, tokenizer, batch_size, padding)

    # Load metric, model, optimizer, and learning rate scheduler.
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    metric.compute = functools.partial(combine_compute, metric)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["task"]["num_labels"])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model"])

    model, dataloader = accelerator.prepare(
        model, dataloader
    )

    logger.info(f" Device: {device}.")
    logger.info(" Starting evaluating model on the data.")
    eval_loss, eval_accuracy, predictions, cm, logits, soft_references = evaluate_data(model, dataloader, metric, device, loss_type)
    logger.info(f" Average Loss: {eval_loss}, Average Accuracy: {eval_accuracy}")

    if "output_predictions" in config["pipeline"]:
        p = Path(config["pipeline"]["output_predictions"]).parent
        p.mkdir(exist_ok=True, parents=True)

        with open(config["pipeline"]["output_predictions"], "w") as f:
            save_dict = {
                "confusion_matrix": cm.tolist(),
                "predictions": predictions,
                "average_loss": float(eval_loss),
                "results": eval_accuracy,
                "logits": logits.tolist(),
                "soft_references": soft_references.tolist(),
            }
            json.dump(save_dict, f)


if __name__ == "__main__":
    main()
