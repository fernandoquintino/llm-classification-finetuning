"""A collection of custom plotting functions.

Functions:
    plot_results: Plots graphs of loss and accuracy during training.
    plot_confusion_matrix: Plots the confusion matrix.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_results(results: dict) -> None:
    """Plots graphs of loss and accuracy during training.

    Args:
        results: The dictionary returned from train_model() or
            train_classifier_model().
    """
    epochs = range(1, len(results["loss_train_history"]) + 1)

    # Loss history
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        epochs,
        results["loss_train_history"],
        marker="o",
        color="blue",
        label="Train"
    )
    ax.plot(
        epochs,
        results["loss_val_history"],
        marker="o",
        color="red",
        label="Val"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()

    # Accuracy history
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        epochs,
        results["acc_train_history"],
        marker="o",
        color="blue",
        label="Train"
    )
    ax.plot(
        epochs,
        results["acc_val_history"],
        marker="o",
        color="red",
        label="Val"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        display_labels: list[str],
    ) -> None:
    """Plots the confusion matrix.

    Args:
        all_labels: 1-d array containing the labels.
        all_preds: 1-d array containing the class predictions.
        display_labels: A list of strings containing the class names.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_labels,
        all_preds,
        display_labels=display_labels,
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Confusion Matrix")
    plt.show()