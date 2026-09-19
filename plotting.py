import matplotlib.pyplot as plt


def plot_results(results: dict) -> None:
    """
    Generates two plots (loss and accuracy history during training) for
    train/val sets.

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