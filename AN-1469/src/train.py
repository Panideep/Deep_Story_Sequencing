import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import pandas as pd

from model import build_model
from utils import create_tf_datasets


def main():

    config = yaml.safe_load(open("config.yaml"))

    train_ds, val_ds = create_tf_datasets(config)

    model = build_model(config)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config["training"]["learning_rate"]
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"]
    )

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]

    results = pd.DataFrame({
        "Train Loss":[train_loss],
        "Val Loss":[val_loss],
        "Val Accuracy":[val_acc]
    })

    results.to_csv("results/tables/results.csv", index=False)

    plt.figure()

    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.savefig("results/figures/loss_curve.png")

    plt.close()

    model.save("results/model.h5")


if __name__ == "__main__":
    main()