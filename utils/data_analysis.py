# %%
acousticstream  = {
    "train_loss": [0.6914, 0.4430, 0.2418, 0.1275, 0.0724, 0.0570, 0.0361, 0.0294, 0.0283, 0.0274],
    "val_loss":   [0.7085, 0.5492, 0.4144, 0.3025, 0.2628, 0.3897, 0.3074, 0.2332, 0.2301, 0.2506],
    "val_acc":    [0.5006, 0.7381, 0.8619, 0.8960, 0.9093, 0.8603, 0.9015, 0.9270, 0.9286, 0.9231]
}

biostream = {
    "train_loss": [0.4005, 0.2597, 0.2077, 0.1788, 0.1626, 0.1530, 0.1461, 0.1402, 0.1353, 0.1322],
    "val_loss":   [0.3153, 0.2803, 0.1718, 0.1778, 0.1690, 0.1663, 0.1509, 0.1607, 0.1313, 0.1497],
    "val_acc":    [0.8590, 0.8771, 0.9318, 0.9293, 0.9329, 0.9317, 0.9412, 0.9351, 0.9492, 0.9438]
    }

dualstream  = {
    "train_loss": [0.6077, 0.2251, 0.0873, 0.0409, 0.0202, 0.0120, 0.0155, 0.0113, 0.0099, 0.0046],
    "val_loss":   [0.5571, 0.6900, 0.1769, 0.0117, 0.0162, 0.0521, 0.0040, 0.0026, 0.0024, 0.0028],
    "val_acc":    [0.7364, 0.7725, 0.9503, 0.9964, 0.9954, 0.9861, 0.9988, 0.9991, 0.9993, 0.9993]
}


# %%
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# metrics_list = [(name, metrics_dict), ...]
def plot_metrics(metrics_list: List[Tuple[str, Dict]], title: str):
    epochs = len(metrics_list[0][1]["train_loss"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for name, metrics in metrics_list:
        axes[0].plot(range(1, epochs + 1), metrics["train_loss"], label=name)
        axes[1].plot(range(1, epochs + 1), metrics["val_loss"], label=name)
        axes[2].plot(range(1, epochs + 1), metrics["val_acc"], label=name)

    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    axes[2].set_title("Validation Accuracy")

    for ax in axes:
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    axes[2].set_xlabel("Epoch")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# %%
plot_metrics(
    [
        ("Acoustic Stream", acousticstream),
        ("Biological Stream", biostream),
        ("Dual Stream", dualstream),
    ],
    "Model Performance Metrics"
)

# %%
