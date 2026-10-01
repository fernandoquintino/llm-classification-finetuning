"""Functions for training or evaluation of models.

Functions:
    train_epoch: Trains the model for one epoch.
    validate_epoch: Validates the model for one epoch.
    train_model: Trains the model for the specified epochs.
    train_classifier_epoch: Trains the classifier for one epoch.
    validate_classifier_epoch: Validates the classifier for one epoch.
    train_classifier_model: Trains the classifier for the specified
        epochs.
    set_seed: Seeds Python's random and PyTorch.
    tracking_start: Resets cuda peak memory stats and starts a timer.
    tracking_stop: Prints peak memory stats on GPUs and duration of the
        execution.
    evaluate_performance: Evaluates performance.
"""

import copy
import gc
import random
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import DeepPreferenceClassifier, PreferenceModel


def train_epoch(
        model: PreferenceModel,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        scaler: torch.amp.GradScaler
) -> tuple[float, float]:
    """Trains the model for one epoch.

    Args:
        model: The model that takes the concatenated
            prompt/response_0/response_1 sequence and its mask and then
            outputs the logits.
        loader: The data loader for CrossEncoderDataset.
        device: The device to process the calculations (e.g., "cuda").
        optimizer: The optimizer.
        loss: The loss function.
        scaler: Scales the loss to prevent fp16 underflow.

    Returns:
        epoch_loss: The average loss for the epoch.
        epoch_acc: The accuracy for the epoch.
    """
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for batch in tqdm(loader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
            )
            batch_loss = loss(output, batch["label"])
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Since last batch may be a different size, we multiply each
        # batch loss by the batch size to get a sum, so the final
        # division by total averages correctly.
        running_loss += batch_loss.item() * batch["label"].size(0)
        predicted = torch.argmax(output, 1)
        total += batch["label"].size(0)
        correct += (predicted == batch["label"]).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    return epoch_loss, epoch_acc


def validate_epoch(
        model: PreferenceModel,
        loader: DataLoader,
        device: torch.device,
        loss: nn.Module,
) -> tuple[float, float]:
    """Validates the model for one epoch.

    Args:
        model: The model that takes the concatenated
            prompt/response_0/response_1 sequence and its mask and then
            outputs the logits.
        loader: The data loader for CrossEncoderDataset.
        device: The device to process the calculations (e.g., "cuda").
        loss: The loss function.

    Returns:
        epoch_loss: The average loss for the epoch.
        epoch_acc: The accuracy for the epoch.
    """
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                batch_loss = loss(output, batch["label"])

            # Since last batch may be a different size, we multiply each
            # batch loss by the batch size to get a sum, so the final
            # division by total averages correctly.
            running_loss += batch_loss.item() * batch["label"].size(0)
            predicted = torch.argmax(output, 1)
            total += batch["label"].size(0)
            correct += (predicted == batch["label"]).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    return epoch_loss, epoch_acc


def train_model(
        model: PreferenceModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        num_epochs: int,
        scaler: torch.amp.GradScaler,
        run_name: str | None = None,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict:
    """Trains the model for the specified epochs.

    mlflow is used to track the runs.

    Args:
        model: The model that takes the concatenated
            prompt/response_0/response_1 sequence and its mask and then
            outputs the logits.
        train_loader: The train loader for CrossEncoderDataset.
        val_loader: The val loader for CrossEncoderDataset.
        device: The device to process the calculations (e.g., "cuda").
        optimizer: The optimizer.
        loss: The loss function.
        num_epochs: The number of epochs to train the model.
        scaler: Scales the loss to prevent fp16 underflow.
        run_name: The name of the run. Defaults to None.
        scheduler: Updates the learning rate based upon val loss.
            Defaults to None.

    Returns:
        dict: A dictionary containing:
            - loss_train_history (list[float]): The history of the
                training loss per epoch.
            - acc_train_history (list[float]): The history of the
                training accuracy per epoch.
            - loss_val_history (list[float]): The history of the val
                loss per epoch.
            - acc_val_history (list[float]): The history of the val
                accuracy per epoch.
            - best_model (PreferenceModel): The model with the best loss
                on the validation set.
            - best_loss (float): The best loss on the validation set.
            - best_model_acc (float): The accuracy of the best_model.
    """
    loss_train_history = []
    acc_train_history = []
    loss_val_history = []
    acc_val_history = []
    best_loss = float("inf")
    best_model_acc = 0.0
    best_model = copy.deepcopy(model)
    best_epoch = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("learning_rate", optimizer.param_groups[0]["lr"])
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("optimizer_type", type(optimizer).__name__)
        if scheduler is not None:
            mlflow.log_param("scheduler_type", type(scheduler).__name__)
            mlflow.log_params(scheduler.state_dict())
        mlflow.log_param("model_type", type(model).__name__)

        print(f"Training on device {device} for {num_epochs} epochs:")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        epoch_bar = tqdm(range(num_epochs), desc="Epochs")
        for epoch in epoch_bar:
            epoch_train_loss, epoch_train_acc = train_epoch(
                model,
                train_loader,
                device,
                optimizer,
                loss,
                scaler
            )
            loss_train_history.append(epoch_train_loss)
            acc_train_history.append(epoch_train_acc)

            epoch_val_loss, epoch_val_acc = validate_epoch(
                model,
                val_loader,
                device,
                loss
            )
            loss_val_history.append(epoch_val_loss)
            acc_val_history.append(epoch_val_acc)

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_acc", epoch_val_acc, step=epoch)
            mlflow.log_metric(
                "learning_rate",
                optimizer.param_groups[0]["lr"],
                step=epoch
            )

            if scheduler is not None:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                if isinstance(model, nn.DataParallel):
                    best_model = copy.deepcopy(model.module)
                else:
                    best_model = copy.deepcopy(model)
                best_model_acc = epoch_val_acc
                best_loss = epoch_val_loss
                mlflow.pytorch.log_state_dict(
                    best_model.state_dict(),
                    artifact_path="best_model_state_dict"
                )
                best_epoch = epoch + 1

            epoch_bar.set_description(
                f"Epoch {epoch + 1} | Train loss: {epoch_train_loss:.4f} acc: "
                f"{epoch_train_acc:.2f}% | Val loss: {epoch_val_loss:.4f} acc:"
                f" {epoch_val_acc:.2f}%"
            )

        mlflow.log_metric("best_model_val_acc", best_model_acc)
        mlflow.log_metric("best_val_loss", best_loss)

    print("Training Complete.")
    print(f"Best Val loss: {best_loss}")
    print(f"Val Acc: {best_model_acc}")
    print(f"Best epoch: {best_epoch}")

    return {
        "loss_train_history": loss_train_history,
        "acc_train_history": acc_train_history,
        "loss_val_history": loss_val_history,
        "acc_val_history": acc_val_history,
        "best_model_acc": best_model_acc,
        "best_model": best_model,
        "best_loss": best_loss,
    }


def train_classifier_epoch(
        model: DeepPreferenceClassifier,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        scaler: torch.amp.GradScaler
) -> tuple[float, float]:
    """Trains the classifier for one epoch.

    Args:
        model: The classifier that takes encoded input and outputs the
            logits.
        loader: The data loader for EncodedPreferenceDataset.
        device: The device to process the calculations (e.g., "cuda").
        optimizer: The optimizer.
        loss: The loss function.
        scaler: Scales the loss to prevent fp16 underflow.

    Returns:
        epoch_loss: The average loss for the epoch.
        epoch_acc: The accuracy for the epoch.
    """
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for batch in tqdm(loader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = model(
                batch["encoded"],
            )
            batch_loss = loss(output, batch["label"])
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Since last batch may be a different size, we multiply each
        # batch loss by the batch size to get a sum, so the final
        # division by total averages correctly.
        running_loss += batch_loss.item() * batch["label"].size(0)
        predicted = torch.argmax(output, 1)
        total += batch["label"].size(0)
        correct += (predicted == batch["label"]).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    return epoch_loss, epoch_acc


def validate_classifier_epoch(
        model: DeepPreferenceClassifier,
        loader: DataLoader,
        device: torch.device,
        loss: nn.Module,
) -> tuple[float, float]:
    """Validates the classifier for one epoch.

    Args:
        model: The classifier that takes encoded input and outputs the
            logits.
        loader: The data loader for EncodedPreferenceDataset.
        device: The device to process the calculations (e.g., "cuda").
        loss: The loss function.

    Returns:
        epoch_loss: The average loss for the epoch.
        epoch_acc: The accuracy for the epoch.
    """
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(
                    batch["encoded"],
                )
                batch_loss = loss(output, batch["label"])

            # Since last batch may be a different size, we multiply each
            # batch loss by the batch size to get a sum, so the final
            # division by total averages correctly.
            running_loss += batch_loss.item() * batch["label"].size(0)
            predicted = torch.argmax(output, 1)
            total += batch["label"].size(0)
            correct += (predicted == batch["label"]).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    return epoch_loss, epoch_acc


def train_classifier_model(
        model: DeepPreferenceClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        num_epochs: int,
        scaler: torch.amp.GradScaler,
        run_name: str | None = None,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict:
    """Trains the classifier for the specified epochs.

    mlflow is used to track the runs.

    Args:
        model: The classifier that takes encoded input and outputs the
            logits.
        train_loader: The train loader for EncodedPreferenceDataset.
        val_loader: The val loader for EncodedPreferenceDataset.
        device: The device to process the calculations (e.g., "cuda").
        optimizer: The optimizer.
        loss: The loss function.
        num_epochs: The number of epochs to train the model.
        scaler: Scales the loss to prevent fp16 underflow.
        run_name: The name of the run. Defaults to None.
        scheduler: Updates the learning rate based upon val loss.
            Defaults to None.

    Returns:
        dict: A dictionary containing:
            - loss_train_history (list[float]): The history of the
                training loss per epoch.
            - acc_train_history (list[float]): The history of the
                training accuracy per epoch.
            - loss_val_history (list[float]): The history of the val
                loss per epoch.
            - acc_val_history (list[float]): The history of the val
                accuracy per epoch.
            - best_model (DeepPreferenceClassifier): The model with the
                best loss on the validation set.
            - best_loss (float): The best loss on the validation set.
            - best_model_acc (float): The accuracy of the best_model.
    """
    loss_train_history = []
    acc_train_history = []
    loss_val_history = []
    acc_val_history = []
    best_loss = float("inf")
    best_model_acc = 0.0
    best_model = copy.deepcopy(model)
    best_epoch = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("learning_rate", optimizer.param_groups[0]["lr"])
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("optimizer_type", type(optimizer).__name__)
        if scheduler is not None:
            mlflow.log_param("scheduler_type", type(scheduler).__name__)
            mlflow.log_params(scheduler.state_dict())
        mlflow.log_param("model_type", type(model).__name__)

        print(f"Training on device {device} for {num_epochs} epochs:")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        epoch_bar = tqdm(range(num_epochs), desc="Epochs")
        for epoch in epoch_bar:
            epoch_train_loss, epoch_train_acc = train_classifier_epoch(
                model,
                train_loader,
                device,
                optimizer,
                loss,
                scaler
            )
            loss_train_history.append(epoch_train_loss)
            acc_train_history.append(epoch_train_acc)

            epoch_val_loss, epoch_val_acc = validate_classifier_epoch(
                model,
                val_loader,
                device,
                loss
            )
            loss_val_history.append(epoch_val_loss)
            acc_val_history.append(epoch_val_acc)

            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_train_acc, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_acc", epoch_val_acc, step=epoch)
            mlflow.log_metric(
                "learning_rate",
                optimizer.param_groups[0]["lr"],
                step=epoch
            )

            if scheduler is not None:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                if isinstance(model, nn.DataParallel):
                    best_model = copy.deepcopy(model.module)
                else:
                    best_model = copy.deepcopy(model)
                best_model_acc = epoch_val_acc
                best_loss = epoch_val_loss
                mlflow.pytorch.log_state_dict(
                    best_model.state_dict(),
                    artifact_path="best_model_classifier_state_dict"
                )
                best_epoch = epoch + 1

            epoch_bar.set_description(
                f"Epoch {epoch + 1} | Train loss: {epoch_train_loss:.4f} acc: "
                f"{epoch_train_acc:.2f}% | Val loss: {epoch_val_loss:.4f} acc:"
                f" {epoch_val_acc:.2f}%"
            )

        mlflow.log_metric("best_model_val_acc", best_model_acc)
        mlflow.log_metric("best_val_loss", best_loss)

    print("Training Complete.")
    print(f"Best Val loss: {best_loss}")
    print(f"Val Acc: {best_model_acc}")
    print(f"Best epoch: {best_epoch}")

    return {
        "loss_train_history": loss_train_history,
        "acc_train_history": acc_train_history,
        "loss_val_history": loss_val_history,
        "acc_val_history": acc_val_history,
        "best_model_acc": best_model_acc,
        "best_model": best_model,
        "best_loss": best_loss,
    }


def set_seed(seed: int) -> torch.Generator:
    """Seeds Python's random and PyTorch.

    Seeds Python's random and PyTorch (CPU, CUDA, MPS) for
    reproducibility. Seeds global state, so call it before each
    training run.

    Args:
        seed: The integer to seed.

    Returns:
        A seeded generator. Pass it to DataLoader(generator=...) to make
        shuffling and the worker's response swaps reproducible.
    """
    random.seed(seed) # Used to permute responses.
    torch.manual_seed(seed) # Used to initialize weights and for dropout
    torch.cuda.manual_seed_all(seed) # Same as above but for GPU
    torch.mps.manual_seed(seed) # Same as above but for MPS
    generator = torch.Generator().manual_seed(seed)
    return generator


def tracking_start() -> float:
    """Starts tracking peak GPU memory and starts a timer.

    Resets PyTorch's peak memory stats for all GPUs and returns the
    time so that when tracking_stop() is called, the peak allocated
    memory and execution time can be reported.

    Returns:
        float: The current time.
    """
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device=i)
    return time.perf_counter()


def tracking_stop(start: float) -> None:
    """Prints peak memory stats on GPUs and duration of the execution.

    Reports PyTorch's peak allocated memory per GPU for machines with
    CUDA.

    Args:
        start: The time the execution began from tracking_start().
    """
    elapsed = time.perf_counter() - start
    print(f"Duration of Execution: {elapsed / 60:.1f} minutes")
    for i in range(torch.cuda.device_count()):
        try:
            peak = torch.cuda.max_memory_allocated(device=i) / 1e9
            print(f"GPU {i} peak memory: {peak:.2f} GB")
        except RuntimeError as e:
            print(f"Could not read memory for GPU {i}: {e}")


def evaluate_performance(
        model: PreferenceModel,
        loader: DataLoader,
        device: torch.device,
        loss: nn.Module,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Evaluates performance.

    Returns the labels with their predictions and the epoch loss and
    accuracy.

    Args:
        model: The model that takes the concatenated
            prompt/response_0/response_1 sequence and its mask and then
            outputs the logits.
        loader: The data loader for CrossEncoderDataset.
        device: The device to process the calculations (e.g., "cuda").
        loss: The loss function.

    Returns:
        all_labels: 1-d array containing the labels.
        all_preds: 1-d array containing the class predictions.
        epoch_loss: The average loss for the epoch.
        epoch_acc: The accuracy for the epoch.
    """
    all_labels = []
    all_preds = []
    running_loss = 0.0
    correct = 0
    total = 0

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                batch_loss = loss(output, batch["label"])

            # Since last batch may be a different size, we multiply each
            # batch loss by the batch size to get a sum, so the final
            # division by total averages correctly.
            running_loss += batch_loss.item() * batch["label"].size(0)
            predicted = torch.argmax(output, 1)
            total += batch["label"].size(0)
            correct += (predicted == batch["label"]).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(batch["label"].cpu())

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_labels, all_preds, epoch_loss, epoch_acc