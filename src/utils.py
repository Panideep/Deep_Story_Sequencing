import tensorflow as tf
import numpy as np
from datasets import load_dataset
from PIL import Image


def preprocess_image(img, size):

    if not isinstance(img, Image.Image):
        img = Image.open(img)

    img = img.resize((size, size))
    # convert to float32 immediately to reduce memory use and avoid default float64
    img = np.asarray(img, dtype=np.float32) / 255.0

    return img


def load_dataset_images(config):

    dataset = load_dataset(config["dataset"]["name"], split="train")

    images = []
    labels = []

    size = config["dataset"]["image_size"]

    for story in dataset:

        story_images = story["images"]

        for position in range(5):

            img = preprocess_image(story_images[position], size)

            images.append(img)
            labels.append(position)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def create_tf_datasets(config):

    images, labels = load_dataset_images(config)

    train_size = int(len(images) * config["training"]["train_split"])

    x_train = images[:train_size]
    y_train = labels[:train_size]

    x_val = images[train_size:]
    y_val = labels[train_size:]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.shuffle(1000).batch(config["training"]["batch_size"])
    val_ds = val_ds.batch(config["training"]["batch_size"])

    return train_ds, val_ds